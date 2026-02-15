import os
import json
import re
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

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

def is_strict(user_request: str) -> bool:
    t = (user_request or "").lower()
    return any(k in t for k in STRICT_TRIGGERS)


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
def run_agent(llm, llm_no_tools, chat_history, user_request) -> str:
    # Plan with the NO-TOOLS model (keeps planning smart + deterministic)
    plan = plan_search(llm_no_tools, user_request)

    chat_history.append(
        HumanMessage(
            content=(
                f"{user_request}\n\n"
                f"(planner intent: {plan['intent']})\n"
                f"(planner queries: {plan['queries']})\n"
                f"(planner notes: {plan['notes']})\n"
            )
        )
    )
    
    if is_strict(user_request):
        chat_history.append(SystemMessage(
        content=(
            "STRICT MODE: The user requested exact code.\n"
            "You MUST call tools and MUST open_code on the relevant implementation before answering.\n"
            "Do NOT answer from summaries or guesses.\n"
            "If you cannot find the implementation, you MUST say what you tried and ask ONE targeted question.\n"
        )
    ))


    tool_registry = {
        "search_symbols_multi": search_symbols_multi,
        "open_code": open_code,
        "find_in_files": find_in_files,
        "trace_assignments": trace_assignments,
        "find_call_sites": find_call_sites,
        "extract_call_arguments": extract_call_arguments,
        "reload_index": reload_index,
    }

    for _ in range(10):
        ai_msg = llm.invoke(chat_history)
        chat_history.append(ai_msg)

        if not ai_msg.tool_calls:
            parsed = _extract_json_object(ai_msg.content)
            if parsed:
                answer = str(parsed.get("answer", "")).strip()
                evidence = parsed.get("evidence", [])
                if isinstance(evidence, list) and evidence:
                    ev = "\n".join(f"- {e}" for e in evidence[:10])
                    return f"{answer}\n\nEvidence:\n{ev}"
                return answer or ai_msg.content
            return ai_msg.content

        for call in ai_msg.tool_calls:
            name = call["name"]
            if name not in tool_registry:
                raise RuntimeError(f"Unexpected tool call: {name}")

            # Defensive: args sometimes arrive as JSON string
            args = call.get("args", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            tool_fn = tool_registry[name]
            result = tool_fn.invoke(args)

            # -------------------
            # Compact tool outputs
            # -------------------
            if name == "search_symbols_multi":
                result = format_hits(result, max_items=12)

            if name == "find_in_files":
                lines = []
                for r in (result or [])[:25]:
                    lines.append(f"{r.get('file')}:{r.get('lineno')}  {r.get('line')}")
                result = "\n".join(lines) if lines else "(no matches)"

            if name == "trace_assignments":
                lines = []
                for r in (result or [])[:20]:
                    lines.append(
                        f"{r.get('file')}:{r.get('lineno')}  [{r.get('kind')}]  {r.get('snippet')}"
                    )
                result = "\n".join(lines) if lines else "(no assignments found)"

            if name == "find_call_sites":
                lines = []
                for r in (result or [])[:25]:
                    lines.append(f"{r.get('file')}:{r.get('lineno')}  {r.get('snippet')}")
                result = "\n".join(lines) if lines else "(no call sites found)"

            if name == "extract_call_arguments":
                if isinstance(result, dict) and "error" not in result:
                    args_s = ", ".join(result.get("args", []))
                    kwargs_s = ", ".join(
                        [f"{k['name']}={k['value']}" for k in result.get("kwargs", [])]
                    )
                    call_src = result.get("call_src", "")
                    result = (
                        f"{result.get('file')}:{result.get('lineno')}\n"
                        f"call: {call_src}\n"
                        f"args: {args_s}\n"
                        f"kwargs: {kwargs_s}"
                    )
                else:
                    result = str(result)

            chat_history.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # Forced-final (no tools)
    forced = SystemMessage(
        content=(
            "TOOL BUDGET EXHAUSTED. Do NOT call any tools. "
            "Using ONLY the evidence already gathered above, output ONLY valid JSON "
            "with keys: answer (string), evidence (array of file:line strings)."
        )
    )
    chat_history.append(forced)
    final = llm_no_tools.invoke(chat_history)
    chat_history.pop()

    parsed = _extract_json_object(final.content)
    if parsed:
        answer = str(parsed.get("answer", "")).strip()
        evidence = parsed.get("evidence", [])
        if isinstance(evidence, list) and evidence:
            ev = "\n".join(f"- {e}" for e in evidence[:10])
            return f"{answer}\n\nEvidence:\n{ev}"
        return answer or final.content

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
    "   b) locate the module implementation file (likely under Modules\\\\...)\n"
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
        raise RuntimeError("Missing OPENAI_API_KEY in your environment or .env file")

    # LLM that CAN call tools
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

    # LLM that CANNOT call tools (planner + forced final)
    llm_no_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Separate histories = separate "agent memories"
    chat_history_qa = [SYSTEM_CODE_QA]
    chat_history_dbg = [SYSTEM_DEBUGGER]

    print("\nTwo-mode agent ready.\n"
          "- Prefix with 'qa:' or 'code:' to force Code Q&A mode\n"
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
            history = chat_history_dbg if mode == "debug" else chat_history_qa

            answer = run_agent(
                llm=llm,
                llm_no_tools=llm_no_tools,
                chat_history=history,
                user_request=user_request,
            )

            print(f"\n[{mode.upper()}]\n{answer}\n")

        except Exception as e:
            print(f"\n[error] {e}\n")


if __name__ == "__main__":
    main()
