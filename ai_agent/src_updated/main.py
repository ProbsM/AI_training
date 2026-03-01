from multiprocessing import context
import os
import json
import re
from dotenv import load_dotenv
from pathlib import Path
from src_codebase_indexer import INDEX_PATH, build_index
from dataclasses import dataclass, field
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage, AIMessage


from src_codebase_indexer import (
    search_symbols_multi,
    open_code,
    find_in_files,
    trace_assignments,
    find_call_sites,
    extract_call_arguments,
    reload_index,
)

STRICT_TRIGGERS = (
    "code exactly",
    "show me the code",
    "open the code",
    "implementation",
    "step into",
)


MAX_RECENT_MESSAGES = 12


@dataclass
class AgentState:
    system_message: SystemMessage
    summary: str = ""
    recent_messages: List[BaseMessage] = field(default_factory=list)

    def add_user(self, text: str):
        self.recent_messages.append(HumanMessage(content=text))
        self._trim()

    def add_ai(self, text: str):
        self.recent_messages.append(AIMessage(content=text))
        self._trim()

    def add_tool(self, content: str, tool_call_id: str):
        self.recent_messages.append(
            ToolMessage(content=content, tool_call_id=tool_call_id)
        )
        self._trim()

    def _trim(self):
        """
        Trim message window safely.
        Never leave a ToolMessage without its preceding assistant tool_call.
        """

        if len(self.recent_messages) <= MAX_RECENT_MESSAGES:
            return

        trimmed = self.recent_messages[-MAX_RECENT_MESSAGES:]

        # If first message is a ToolMessage,
        # we must prepend its matching assistant tool_call.
        if isinstance(trimmed[0], ToolMessage):
            # walk backward in original list to find preceding assistant
            for i in range(len(self.recent_messages) - MAX_RECENT_MESSAGES - 1, -1, -1):
                candidate = self.recent_messages[i]
                if hasattr(candidate, "tool_calls") and candidate.tool_calls:
                    trimmed.insert(0, candidate)
                    break

        self.recent_messages = trimmed

    def build_context(self) -> List[BaseMessage]:
        context = [self.system_message]

        if self.summary.strip():
            context.append(SystemMessage(
                content="MEMORY SUMMARY:\n" + self.summary
            ))

        context.extend(self.recent_messages)
        return context

    def update_summary(self, llm_no_tools):
        transcript = "\n".join(
            f"{type(m).__name__}: {getattr(m, 'content', '')[:1500]}"
            for m in self.recent_messages
        )

        prompt = (
            "Update this rolling technical memory summary.\n\n"
            "Keep ONLY:\n"
            "- discovered files\n"
            "- key functions/classes\n"
            "- confirmed conclusions\n"
            "- active hypotheses\n"
            "- unresolved issues\n\n"
            "Remove chatter and code blocks.\n\n"
            f"Existing summary:\n{self.summary}\n\n"
            f"New transcript:\n{transcript}\n\n"
            "Return ONLY updated summary text."
        )

        resp = llm_no_tools.invoke(prompt)
        self.summary = resp.content.strip()

MAX_RECENT_MESSAGES = 12  # tune 8–16


MAX_TOOL_CHARS = 6000  # adjust 4000–8000 depending on needs


def compact_tool_output(text: str, max_chars: int = MAX_TOOL_CHARS) -> str:
    """
    Compact large tool outputs safely.

    - Preserve beginning and end
    - Keep line numbers intact
    - Prevent context explosion
    """

    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]

    return (
        head
        + "\n\n... [TRUNCATED FOR CONTEXT CONTROL] ...\n\n"
        + tail
    )



# Per-agent rolling summaries
summary_qa = ""
summary_dbg = ""
def update_summary(llm_no_tools, old_summary: str, recent_messages: list) -> str:
    """
    Compress conversation into durable memory.
    Keep: discovered files, functions, conclusions, unresolved questions.
    Drop: tool noise, verbose snippets.
    """
    
    transcript = "\n".join(
        f"{type(m).__name__}: {getattr(m, 'content', '')[:2000]}"
        for m in recent_messages
    )
    
    prompt = (
        "You are maintaining a compact memory summary for a code analysis agent.\n\n"
        "Update the existing summary using the new transcript.\n\n"
        "Keep ONLY durable technical facts:\n"
        "- Discovered file paths\n"
        "- Important functions/classes\n"
        "- Confirmed conclusions (with evidence)\n"
        "- Active hypotheses\n"
        "- Unresolved questions\n\n"
        "Remove chatter, repetition, and large code blocks.\n\n"
        f"Existing summary:\n{old_summary}\n\n"
        f"New transcript:\n{transcript}\n\n"
        "Return ONLY the updated summary text."
    )
    
    resp = llm_no_tools.invoke(prompt)
    return resp.content.strip()

def build_context(system_msg, summary_text: str, chat_history: list):
    """
    Reconstruct context each turn:
      - system
      - summary
      - recent window
    """
    context = [system_msg]

    if summary_text.strip():
        context.append(SystemMessage(
            content="MEMORY SUMMARY:\n" + summary_text
        ))

    # Keep only last N messages
    recent = chat_history[-MAX_RECENT_MESSAGES:]
    context.extend(recent)

    return context

def is_strict(user_request: str) -> bool:
    t = (user_request or "").lower()
    return any(k in t for k in STRICT_TRIGGERS)


def score_symbol(sym: dict, queries: list[str]) -> int:
    """
    Simple heuristic relevance score.
    Higher score = more relevant.
    """

    score = 0
    name = (sym.get("name") or "").lower()
    qual = (sym.get("qualname") or "").lower()
    doc = (sym.get("docstring") or "").lower()

    for q in queries:
        q = q.lower()

        # Exact name match (strongest)
        if name == q:
            score += 50

        # Name contains query
        if q in name:
            score += 25

        # Qualname contains query
        if q in qual:
            score += 15

        # Docstring mention
        if q in doc:
            score += 5

    return score

# ----------------------------
# Helper formatting / parsing
# ----------------------------
def format_hits(hits: list[dict], max_items: int = 12) -> str:
    seen = set()
    out = []

    for h in hits or []:
        file_ = h.get("file", "")
        lineno = h.get("lineno", "")
        qualname = h.get("qualname", "")
        key = f"{file_}:{lineno}:{qualname}"
        if key in seen:
            continue
        seen.add(key)

        kind = h.get("kind", "")
        sig = h.get("signature", "")
        doc = (h.get("docstring") or "").strip().replace("\n", " ")
        doc = (doc[:180] + "...") if len(doc) > 180 else doc

        if doc:
            out.append(f"- {kind}: {qualname} {sig}\n  {file_}:{lineno}\n  doc: {doc}")
        else:
            out.append(f"- {kind}: {qualname} {sig}\n  {file_}:{lineno}")

        if len(out) >= max_items:
            break

    return "\n".join(out) if out else "(no matches)"

def ensure_index_exists() -> None:
    if not Path(INDEX_PATH).exists():
        # Build it in the directory where INDEX_PATH expects it
        suite_root = Path(INDEX_PATH).parent  
        print(f"[info] code_index.json missing. Rebuilding at: {suite_root}")
        build_index(str(suite_root))

def _extract_json_object(text: str) -> dict:
    """
    Extract a JSON object from a model response.
    Accepts either pure JSON or text containing an embedded {...}.
    """
    text = (text or "").strip()

    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}


def plan_search(llm_planner: ChatOpenAI, user_request: str) -> dict:
    """
    Vague user request -> concrete search queries.
    IMPORTANT: Use a NO-TOOLS LLM for planning to avoid tool-call distractions.
    """
    prompt = (
        "Return ONLY valid JSON (no markdown) with keys:\n"
        "  intent: string\n"
        "  queries: array of 3 to 6 short keyword queries (NOT full sentences)\n"
        "  notes: string\n\n"
        "Guidance:\n"
        "- Prefer symbol names, module names, filenames, config keys.\n"
        "- If user mentions a specific test (e.g. DPIN_EPA_FreqShmoo), include that exact token.\n"
        "- If user asks pass/fail criteria, include queries like: 'limits', 'verify', 'pass', 'fail', 'result', 'validation'.\n\n"
        f"User request: {user_request}"
    )

    resp = llm_planner.invoke(prompt)
    plan = _extract_json_object(getattr(resp, "content", str(resp)))

    queries = plan.get("queries")
    if not isinstance(queries, list):
        queries = [user_request]

    plan_out = {
        "intent": str(plan.get("intent", "")).strip(),
        "notes": str(plan.get("notes", "")).strip(),
        "queries": [str(q).strip() for q in queries if str(q).strip()],
    }
    if not plan_out["queries"]:
        plan_out["queries"] = [user_request]

    return plan_out

def ensure_valid_json(llm_no_tools, context, raw_text):
    """
    Ensure the model returns valid JSON with keys:
    answer (string)
    evidence (list)
    """

    parsed = _extract_json_object(raw_text)
    if parsed:
        return parsed

    repair_prompt = (
        "Your previous response was not valid JSON.\n\n"
        "Return ONLY valid JSON with keys:\n"
        "{ \"answer\": string, \"evidence\": [string, ...] }\n\n"
        "Do not include markdown or commentary.\n\n"
        f"Previous response:\n{raw_text}"
    )

    repaired = llm_no_tools.invoke(context + [SystemMessage(content=repair_prompt)])
    parsed = _extract_json_object(repaired.content)

    return parsed if parsed else None


# ----------------------------
# Router (QA vs Debug)
# ----------------------------
DEBUG_KEYWORDS = {
    "error", "exception", "traceback", "stack", "crash", "fails", "failing",
    "bug", "debug", "fix", "why doesn't", "not working", "hang", "timeout",
}

def choose_agent(user_text: str) -> str:
    """
    Returns: 'qa' or 'debug'
    Manual override:
      - "debug: ..." or "dbg: ..." forces debug agent
      - "qa: ..." or "code: ..." forces qa agent
    Otherwise uses simple keyword heuristics.
    """
    t = (user_text or "").strip()
    tl = t.lower()

    if tl.startswith("debug:") or tl.startswith("dbg:"):
        return "debug"
    if tl.startswith("qa:") or tl.startswith("code:"):
        return "qa"

    if any(k in tl for k in DEBUG_KEYWORDS):
        return "debug"

    return "qa"


# ----------------------------
# Agent loop (shared engine)
# ----------------------------
def run_agent(llm, llm_no_tools, state: AgentState, user_request: str) -> str:
    strict_mode = is_strict(user_request)
    open_code_used = False

    tool_registry = {
        "search_symbols_multi": search_symbols_multi,
        "open_code": open_code,
        "find_in_files": find_in_files,
        "trace_assignments": trace_assignments,
        "find_call_sites": find_call_sites,
        "extract_call_arguments": extract_call_arguments,
        "reload_index": reload_index,
    }

    # -----------------------------------------
    # 🔹 Deterministic Initial Search Seeding
    # -----------------------------------------
    plan = plan_search(llm_no_tools, user_request)

    if plan.get("queries"):
        try:
            seed_hits = search_symbols_multi.invoke({
                "queries": plan["queries"]
            })

            # 🔥 Rank by relevance
            seed_hits = sorted(
                seed_hits,
                key=lambda s: score_symbol(s, plan["queries"]),
                reverse=True
            )

            # Keep only top N
            seed_hits = seed_hits[:12]
            seed_text = format_hits(seed_hits, max_items=8)
            seed_text = compact_tool_output(seed_text, max_chars=4000)
            state.recent_messages.append(
                SystemMessage(
                    content="INITIAL SEARCH RESULTS (seeded):\n" + seed_text
                )
            )
            state._trim()
            
        except Exception:
            pass

    # -----------------------------------------
    # 🔹 Tool Loop
    # -----------------------------------------
    for _ in range(8):
        context = state.build_context()
        ai_msg = llm.invoke(context)

        state.recent_messages.append(ai_msg)

        # -------------------------------------
        # No tool calls → candidate final
        # -------------------------------------
        if not getattr(ai_msg, "tool_calls", None):

            # 🔒 STRICT MODE ENFORCEMENT
            if strict_mode and not open_code_used:
                state.add_ai(
                    "STRICT MODE VIOLATION: You must call open_code before answering."
                )
                continue

            parsed = ensure_valid_json(
                llm_no_tools,
                context,
                ai_msg.content
            )

            if parsed:
                return parsed.get("answer", ai_msg.content)

            return ai_msg.content

        # -------------------------------------
        # Handle Tool Calls
        # -------------------------------------
        for call in ai_msg.tool_calls:
            name = call["name"]
            args = call.get("args", {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if name not in tool_registry:
                raise RuntimeError(f"Unexpected tool call: {name}")

            tool_fn = tool_registry[name]
            result = tool_fn.invoke(args)

            result_str = str(result)

            # 🔥 Track open_code usage
            if name == "open_code":
                open_code_used = True

            # 🔥 Compact large outputs
            if name == "open_code":
                result_str = compact_tool_output(result_str)
            elif name in {
                "find_in_files",
                "trace_assignments",
                "find_call_sites",
            }:
                result_str = compact_tool_output(result_str, max_chars=4000)

            state.add_tool(result_str, call["id"])

    # -----------------------------------------
    # 🔹 Forced Final (Tool Budget Exhausted)
    # -----------------------------------------
    forced = SystemMessage(
        content=(
            "Tool budget exhausted. Using ONLY gathered evidence, "
            "output valid JSON with keys: answer and evidence."
        )
    )

    context = state.build_context() + [forced]
    final = llm_no_tools.invoke(context)

    parsed = ensure_valid_json(
        llm_no_tools,
        context,
        final.content
    )

    if parsed:
        return parsed.get("answer", final.content)

    return final.content

# ----------------------------
# Two SystemMessages ("two agents")
# ----------------------------
SYSTEM_CODE_QA = SystemMessage(content=(
    "You are a codebase Q&A assistant for a Python hardware test suite.\n"
    "Goal: help the user understand the codebase (what/where/how) with citations.\n\n"
    "Tools:\n"
    "- search_symbols_multi, open_code, find_in_files, trace_assignments, find_call_sites, extract_call_arguments, reload_index\n\n"
    "Rules:\n"
    "1) Use search_symbols_multi first.\n"
    "2) If asked HOW something works, open_code before explaining.\n"
    "3) If asked where a value/path is set, trace_assignments then open_code.\n"
    "4) If asked where a function is called, find_call_sites then open_code.\n"
    "5) Be concise; no speculation. If you didn't find it, say what you searched.\n\n"
    "If the user asks \"how does <module> work/run\", you MUST open_code the module entrypoint and cite the relevant lines.\n"
    "FINAL RESPONSE MUST BE ONLY valid JSON:\n"
    "{ \"answer\": \"...\", \"evidence\": [\"file:line\", \"file:line\"] }\n"
))

SYSTEM_DEBUGGER = SystemMessage(content=(
    "You are a debugging assistant for a Python hardware test suite.\n"
    "Goal: diagnose failures and propose fixes grounded in code evidence.\n\n"
    "Tools:\n"
    "- search_symbols_multi, open_code, find_in_files, trace_assignments, find_call_sites, extract_call_arguments, reload_index\n\n"
    "Rules:\n"
    "0) NO SPECULATION RULE:\n"
    "   If the user asks \"is something wrong?\" but provides no symptom (error, failing behavior, unexpected output),\n"
    "   you MUST NOT guess. Ask ONE targeted question for the missing symptom/traceback/log.\n"
    "1) Restate the symptom and identify likely failure points.\n"
    "2) For errors/failures, you MUST open_code around the suspected code path.\n"
    "3) Trace call chain and arguments: find_call_sites -> extract_call_arguments.\n"
    "4) Trace values: trace_assignments -> open_code around assignments.\n"
    "5) Provide a minimal fix and explain why it fixes root cause.\n"
    "6) If essential info (traceback/log) is missing, ask ONE targeted question.\n"
    "7) Module entrypoint rule:\n"
    "   For questions like: \"how does <module_name> run?\" or \"how is <module_name> executed?\"\n"
    "   Do NOT assume flow control.\n"
    "   You MUST do ALL of the following before answering:\n"
    "   a) find_in_files(\"<module_name>\") (also try common variations like 'Calib', 'Calibration', etc.)\n"
    "   b) locate the module implementation file (likely under SRCLooping\\\\Modules\\\\...)\n"
    "   c) open_code the entrypoint (Run/execute/main-like function) and cite it\n"
    "   d) find_call_sites(entrypoint_name) and open_code the most relevant call site\n\n"
    "8) If you have not opened code relevant to the claim, you may not claim a bug.\n"
    "FINAL RESPONSE MUST BE ONLY valid JSON:\n"
    "{ \"answer\": \"...\", \"evidence\": [\"file:line\", \"file:line\"] }\n"
))



# ----------------------------
# Main entrypoint
# ----------------------------
def main() -> None:
    load_dotenv(override=True)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY")

    ensure_index_exists()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(
        [
            search_symbols_multi,
            open_code,
            find_in_files,
            trace_assignments,
            find_call_sites,
            extract_call_arguments,
            reload_index,
        ]
    )

    llm_no_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    qa_state = AgentState(system_message=SYSTEM_CODE_QA)
    dbg_state = AgentState(system_message=SYSTEM_DEBUGGER)

    print("\nTwo-mode agent ready.\n"
          "- Prefix with 'qa:' or 'code:' to force QA mode\n"
          "- Prefix with 'debug:' or 'dbg:' to force Debug mode\n"
          "Type 'exit' to quit.\n")

    while True:
        user_request = input("Find> ").strip()

        if not user_request:
            continue

        if user_request.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            mode = choose_agent(user_request)
            state = dbg_state if mode == "debug" else qa_state

            state.add_user(user_request)

            answer = run_agent(
                llm=llm,
                llm_no_tools=llm_no_tools,
                state=state,
                user_request=user_request,
            )

            print(f"\n[{mode.upper()}]\n{answer}\n")

            state.add_ai(answer)

            state.update_summary(llm_no_tools)

        except Exception as e:
            print(f"\n[error] {e}\n")

if __name__ == "__main__":
    main()
