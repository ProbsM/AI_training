from multiprocessing import context
import os
import json
import re
from dotenv import load_dotenv
from pathlib import Path
from src_codebase_indexer import INDEX_PATH, build_index
from dataclasses import dataclass, field
from typing import Optional, List

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
    find_files,
)

STRICT_TRIGGERS = (
    "code exactly",
    "show me the code",
    "open the code",
    "implementation",
    "step into",
)

DEBUG_KEYWORDS = {
    "error", "exception", "traceback", "stack", "crash", "fails", "failing",
    "bug", "debug", "fix", "wrong", "issue", "problem", "syntax",
    "why doesn't", "not working", "hang", "timeout",
}




MAX_RECENT_MESSAGES = 12  # tune 8–16
MAX_TOOL_CHARS = 6000  # adjust 4000–8000 depending on needs

@dataclass
class AgentState:
    system_message: SystemMessage
    summary: str = ""
    recent_messages: List[BaseMessage] = field(default_factory=list)

    # ✅ Focus lock to prevent “wrong module” drift
    focus_file: Optional[str] = None
    
    # in AgentState
    focus_module: str | None = None

    def set_focus_module(self, module_name: str | None):
        self.focus_module = module_name.lower().strip() if module_name else None

    def add_user(self, text: str):
        self.recent_messages.append(HumanMessage(content=text))
        self._trim()

    def add_ai(self, text: str):
        self.recent_messages.append(AIMessage(content=text))
        self._trim()

    def add_tool(self, content: str, tool_call_id: str):
        # ToolMessage must ONLY be used as a response to a real assistant tool_call
        self.recent_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
        self._trim()

    def set_focus(self, file_path: Optional[str]):
        self.focus_file = file_path
        if file_path:
            # Put the lock into context so the model can’t "wander"
            self.recent_messages.append(SystemMessage(
                content=f"FOCUS FILE LOCK: {file_path}\n"
                        f"You MUST use this file for analysis until the user changes the target file."
            ))
            self._trim()

    def build_context(self) -> List[BaseMessage]:
        context: List[BaseMessage] = [self.system_message]
        if self.focus_module:
            context.append(SystemMessage(
                content=f"ACTIVE MODULE: {self.focus_module}\n"
                        f"Do NOT use other modules unless the user changes the target."
            ))

        if self.summary.strip():
            context.append(SystemMessage(content="MEMORY SUMMARY:\n" + self.summary))

        if self.focus_file:
            context.append(SystemMessage(
                content=f"ACTIVE FILE: {self.focus_file}\n"
                        f"Do NOT switch files unless the user asks to."
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
            "Keep ONLY durable technical facts:\n"
            "- discovered file paths\n"
            "- key functions/classes\n"
            "- confirmed conclusions (grounded in opened code)\n"
            "- unresolved questions\n\n"
            "Remove chatter, repetition, and large code blocks.\n\n"
            f"Existing summary:\n{self.summary}\n\n"
            f"New transcript:\n{transcript}\n\n"
            "Return ONLY updated summary text."
        )

        resp = llm_no_tools.invoke(prompt)
        self.summary = resp.content.strip()

    def _trim(self):
        """
        Hard window limit, but never leave a ToolMessage without its
        preceding assistant tool_calls message (protocol-safe).
        """
        if len(self.recent_messages) <= MAX_RECENT_MESSAGES:
            return

        trimmed = self.recent_messages[-MAX_RECENT_MESSAGES:]

        # If the first kept message is a ToolMessage, prepend the closest preceding
        # assistant message that contained tool_calls.
        if isinstance(trimmed[0], ToolMessage):
            # search backward in the original list before the trimmed window
            start_idx = len(self.recent_messages) - MAX_RECENT_MESSAGES - 1
            for i in range(start_idx, -1, -1):
                candidate = self.recent_messages[i]
                if getattr(candidate, "tool_calls", None):
                    trimmed.insert(0, candidate)
                    break

        self.recent_messages = trimmed

def infer_module_from_query(user_request: str) -> str | None:
    t = (user_request or "").lower()
    # common pattern: "<module> __init__.py" or "<module> module"
    m = re.search(r"\b([a-z][a-z0-9_]+)\b\s+(__init__\.py|module)\b", t)
    if m:
        return m.group(1)
    return None

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


LINE_REF_RE = re.compile(r"(?P<path>[\w\-.\\\/]+\.py)\s*(?:[: ]line\s*)?(?P<line>\d+)?", re.IGNORECASE)
def _extract_file_and_line(user_text: str):
    m = LINE_REF_RE.search(user_text or "")
    if not m:
        return None, None
    return m.group("path"), (int(m.group("line")) if m.group("line") else None)

def _pick_best_file(matches: list[dict]) -> str | None:
    # Prefer shortest path + contains "ChannelCard" or "Modules" if present
    if not matches:
        return None
    def key(r):
        p = (r.get("file") or "").lower()
        score = 0
        if "channelcard" in p: score -= 50
        if "\\modules\\" in p or "/modules/" in p: score -= 10
        score += len(p)
        return score
    return sorted(matches, key=key)[0].get("file")

def resolve_and_open_seed(state: AgentState, llm_no_tools, user_request: str):
    # bind module if user implies one
    mod = infer_module_from_query(user_request)
    if mod:
        state.set_focus_module(mod)

    # if module focus exists and user mentions __init__.py, search ONLY within that module
    if state.focus_module and "__init__.py" in (user_request or "").lower():
        must = f"\\modules\\{state.focus_module}\\"  # constraint, not hardcoded module name
        candidates = find_files.invoke({"path_query": "__init__.py", "must_contain": must, "max_results": 10})
        if candidates:
            state.set_focus(candidates[0])  # set focus_file
            code = open_code.invoke({"file_path": candidates[0], "start_line": 1, "end_line": 200})
            code = compact_tool_output(code)
            state.recent_messages.append(SystemMessage(content=f"FOCUS FILE CONTENT:\n{code}"))
            state._trim()
            return

    # fallback (no module inferred): your normal symbol-based open
    ...

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
        suite_root = Path(INDEX_PATH).parent  # ...\SRCLoopingSuite_Rev3.1.0.6
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
        "find_files": find_files,  # NEW: must also be in llm.bind_tools(...)
    }

    # Deterministic file resolution + auto-open seeding (SystemMessage injection only)
    resolve_and_open_seed(state, llm_no_tools, user_request)

    for _ in range(8):
        context = state.build_context()
        ai_msg = llm.invoke(context)

        state.recent_messages.append(ai_msg)
        state._trim()

        tool_calls = getattr(ai_msg, "tool_calls", None) or []

        if tool_calls:
            for call in tool_calls:
                name = call.get("name")
                if name not in tool_registry:
                    raise RuntimeError(f"Unexpected tool call: {name}")

                args = call.get("args", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}

                tool_fn = tool_registry[name]
                result = tool_fn.invoke(args)
                result_str = str(result)

                if name == "open_code":
                    open_code_used = True

                # Compact large tool outputs
                if name == "open_code":
                    result_str = compact_tool_output(result_str)
                elif name in {"find_in_files", "trace_assignments", "find_call_sites", "find_files"}:
                    result_str = compact_tool_output(result_str, max_chars=4000)

                state.add_tool(result_str, call["id"])

            continue

        if strict_mode and not open_code_used:
            state.add_ai(
                "STRICT MODE VIOLATION: You must call open_code on the relevant implementation before answering."
            )
            continue

        parsed = ensure_valid_json(llm_no_tools, context, ai_msg.content)
        if parsed:
            return parsed.get("answer", ai_msg.content)

        return ai_msg.content

    forced = SystemMessage(
        content=(
            "TOOL BUDGET EXHAUSTED. Do NOT call any tools. "
            "Using ONLY the evidence already gathered above, output ONLY valid JSON "
            "with keys: answer (string), evidence (array of file:line strings)."
        )
    )

    context = state.build_context() + [forced]
    final = llm_no_tools.invoke(context)

    parsed = ensure_valid_json(llm_no_tools, context, final.content)
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
    "- search_symbols_multi, open_code, find_files, find_in_files, trace_assignments, find_call_sites, extract_call_arguments, reload_index\n\n"
    "Rules:\n"
    "0) If the user mentions a filename/path (e.g. ChannelCard\\__init__.py), use find_files first, then open_code.\n"
    "1) Otherwise use search_symbols_multi first.\n"
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
    "- search_symbols_multi, open_code, find_files, find_in_files, trace_assignments, find_call_sites, extract_call_arguments, reload_index\n\n"
    "Rules:\n"
    "0) NO SPECULATION RULE:\n"
    "   If the user asks \"is something wrong?\" but provides no symptom (error, failing behavior, unexpected output),\n"
    "   you MUST NOT guess. Ask ONE targeted question for the missing symptom/traceback/log.\n"
    "1) If the user mentions a filename/path, use find_files first, then open_code that file.\n"
    "2) For errors/failures, you MUST open_code around the suspected code path.\n"
    "3) Trace call chain and arguments: find_call_sites -> extract_call_arguments.\n"
    "4) Trace values: trace_assignments -> open_code around assignments.\n"
    "5) Provide a minimal fix and explain why it fixes root cause.\n"
    "6) If essential info (traceback/log) is missing, ask ONE targeted question.\n"
    "7) Module entrypoint rule:\n"
    "   For questions like: \"how does <module_name> run?\" or \"how is <module_name> executed?\"\n"
    "   Do NOT assume flow control.\n"
    "   You MUST do ALL of the following before answering:\n"
    "   a) find_files(\"<module_name>\") and/or find_in_files(\"<module_name>\")\n"
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

    # LLM WITH tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(
        [
            search_symbols_multi,
            open_code,
            find_in_files,
            trace_assignments,
            find_call_sites,
            extract_call_arguments,
            reload_index,
            find_files,
        ]
    )

    # LLM WITHOUT tools (planner / summary / repair)
    llm_no_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Independent agent states
    qa_state = AgentState(system_message=SYSTEM_CODE_QA)
    dbg_state = AgentState(system_message=SYSTEM_DEBUGGER)

    print(
        "\nTwo-mode agent ready.\n"
        "- Prefix with 'qa:' or 'code:' to force QA mode\n"
        "- Prefix with 'debug:' or 'dbg:' to force Debug mode\n"
        "Type 'exit' to quit.\n"
    )

    while True:
        try:
            user_request = input("Find> ").strip()

            if not user_request:
                continue

            if user_request.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            mode = choose_agent(user_request)
            state = dbg_state if mode == "debug" else qa_state

            # Add user turn once
            state.add_user(user_request)

            answer = run_agent(
                llm=llm,
                llm_no_tools=llm_no_tools,
                state=state,
                user_request=user_request,
            )

            print(f"\n[{mode.upper()}]\n{answer}\n")

            # Add assistant answer once
            state.add_ai(answer)

            # Update rolling summary
            state.update_summary(llm_no_tools)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[error] {e}\n")

if __name__ == "__main__":
    main()
