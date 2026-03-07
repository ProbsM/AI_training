"""
agent.py
--------
Core chat agent for the Hardware Test Codebase Expert.

Ties together:
  - retriever.py  : fetches relevant code chunks from ChromaDB
  - memory.py     : saves and loads conversation sessions
  - Claude Haiku  : reasons over retrieved chunks and chat history

Supports:
  - Answering questions about the codebase
  - Debugging issues using pasted log outputs
  - Generating new test scripts modeled on existing code
  - Generating new hardware API integrations
  - Saving conversations by name
  - Loading previous conversations from a numbered list

Usage (as a module):
    from agent import CodebaseAgent
    agent = CodebaseAgent()
    response = agent.chat("how does the voltage threshold check work?")

Usage (CLI for quick testing):
    uv run agent.py
"""

import os
import re
from anthropic import Anthropic
from retriever import CodebaseRetriever
import memory as mem


# ── Configuration ─────────────────────────────────────────────────────────────

AGENT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS  = 4096
MAX_HISTORY = 20   # Max conversation turns to keep in context


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert software engineer with deep knowledge of a large, modular Python codebase that exercises and validates hardware instruments through automated testing. Your job is to help both newcomers and experienced engineers work effectively with this codebase.

You have been provided with the most relevant code chunks from the codebase to answer the user's question. Always ground your answers in these retrieved chunks — do not invent function names, variable names, or behavior that you cannot see in the provided code.

## Your Capabilities

**1. Answer questions about the codebase**
- Explain what functions, classes, and modules do in plain English
- Describe how different modules interact with each other
- Clarify how hardware APIs are used in context
- Help newcomers understand the overall structure and patterns

**2. Debug issues using pasted logs**
- When a user pastes log output or error tracebacks, cross-reference them against the retrieved code chunks
- Identify the root cause of failures clearly and specifically
- Suggest a concrete fix, showing exactly what to change and where

**3. Generate new test scripts**
- When asked to write new test scripts, always model them on the patterns, conventions, and style of the existing code chunks provided
- Match variable naming conventions, logging patterns, and pass/fail reporting style of the existing codebase
- Never invent new patterns when existing ones are visible in the retrieved chunks

**4. Generate new hardware API integrations**
- When asked to write new hardware API integrations, follow the same structure and conventions as existing integrations in the retrieved chunks
- Always include proper error handling consistent with the existing codebase style

## Important Rules
- Always cite which file and function your answer is based on (e.g. "In `SRCLooping/Modules/HalInitCheck/hal_check.py`, the `check_voltage()` function...")
- If the retrieved chunks do not contain enough information to answer confidently, say so clearly and suggest the user provide more context or paste relevant log output
- When suggesting code changes, always show the full modified function — never show partial snippets that leave the user guessing
- Do not make assumptions about hardware behavior that you cannot verify from the code
- Keep explanations concise for experienced engineers, but offer to elaborate for newcomers
"""



# ── Intent Detection ──────────────────────────────────────────────────────────

def _detect_save_intent(message: str) -> str | None:
    """
    Detect if the user wants to save the conversation.
    Returns the session name if detected, None otherwise.

    Matches patterns like:
      "save this conversation as voltage_debug"
      "save as hal_issue"
      "save conversation voltage_debug"
    """
    patterns = [
    r"save\s+(?:this\s+)?(?:conv\w*\s+)?as\s+(.+)",
    r"save\s+(?:this\s+)?conv\w*\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message.lower())
        if match:
            return match.group(1).strip()
    return None


def _detect_load_intent(message: str) -> bool:
    """Detect if the user wants to load a saved conversation."""
    keywords = ["load", "restore", "open session", "load conversation", "load session"]
    return any(kw in message.lower() for kw in keywords)


def _detect_load_selection(message: str) -> int | None:
    """
    Detect if the user is selecting a session by number from the list.
    Returns the integer selection if detected, None otherwise.
    """
    stripped = message.strip()
    if stripped.isdigit():
        return int(stripped)
    return None


# ── Agent ─────────────────────────────────────────────────────────────────────

class CodebaseAgent:
    """
    Hardware Test Codebase Expert Agent.

    Maintains conversation history across turns and retrieves relevant
    code chunks from ChromaDB for every user message.

    Memory states:
      - idle            : normal chat
      - awaiting_load   : user asked to load, agent listed sessions, waiting for number selection
    """

    def __init__(self, openai_api_key: str = None, anthropic_api_key: str = None):
        self.anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_key:
            raise ValueError("Missing ANTHROPIC_API_KEY")

        self.client    = Anthropic(api_key=self.anthropic_key)
        self.retriever = CodebaseRetriever(openai_api_key=openai_api_key)
        self.history   = []

        # Memory state tracking
        self._memory_state   = "idle"
        self._session_list   = []   # Holds listed sessions while awaiting selection

        print(f"[agent] Initialized with model: {AGENT_MODEL}")


    def chat(self, user_message: str) -> str:
        """
        Main entry point. Routes the message to the appropriate handler
        based on intent detection before falling through to normal RAG chat.
        """

        # ── Handle session selection (user picking a number from load list) ──
        if self._memory_state == "awaiting_load":
            selection = _detect_load_selection(user_message)
            if selection is not None:
                return self._handle_load_selection(selection)
            else:
                # User typed something other than a number — cancel load flow
                self._memory_state = "idle"
                self._session_list = []
                # Fall through to normal chat

        # ── Detect save intent ────────────────────────────────────────────────
        session_name = _detect_save_intent(user_message)
        if session_name:
            return self._handle_save(session_name)

        # ── Detect load intent ────────────────────────────────────────────────
        if _detect_load_intent(user_message):
            return self._handle_load_list()

        # ── Normal RAG chat ───────────────────────────────────────────────────
        return self._rag_chat(user_message)


    def _rag_chat(self, user_message: str) -> str:
        """Standard RAG chat — retrieve chunks, inject context, call Claude."""
        
        # Retrieve relevant code chunks
        chunks  = self.retriever.retrieve(user_message)
        context = self.retriever.format_chunks_for_prompt(chunks)

        # Inject retrieved context into the user message
        augmented_message = f"""## Retrieved Code Context
The following code chunks were retrieved from the codebase as most relevant to your question:

{context}

---

## User Question
{user_message}"""

        # Append to history
        self.history.append({"role": "user", "content": augmented_message})
        
        
        # DEBUG — remove after checking (how costly a specific session is)
        total_chars = sum(len(m["content"]) for m in self.history)
        print(f"[debug] Sending ~{total_chars // 4} tokens to Claude this turn")

        # Trim history to stay within context window
        if len(self.history) > MAX_HISTORY * 2:
            self.history = self.history[-(MAX_HISTORY * 2):]

        # Call Claude
        response = self.client.messages.create(
            model=AGENT_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=self.history,
        )

        assistant_message = response.content[0].text
        self.history.append({"role": "assistant", "content": assistant_message})

        return assistant_message


    def _handle_save(self, session_name: str) -> str:
        """Save the current conversation history to a JSON file."""
        if not self.history:
            return "There's no conversation to save yet."

        try:
            file_path = mem.save_session(self.history, session_name)
            return f"Conversation saved as **{Path(file_path).stem}**."
        except Exception as e:
            return f"Failed to save conversation: {e}"


    def _handle_load_list(self) -> str:
        """List available saved sessions and enter awaiting_load state."""
        sessions = mem.list_sessions()

        if not sessions:
            return "No saved sessions found."

        self._session_list = sessions
        self._memory_state = "awaiting_load"
        return mem.format_session_list(sessions)


    def _handle_load_selection(self, selection: int) -> str:
        """Load the selected session, summarize if needed, restore history."""
        self._memory_state = "idle"

        restored_history, confirmation = mem.load_session(
            selection, self._session_list, self.client
        )
        self._session_list = []

        if restored_history is None:
            return confirmation  # Error message

        self.history = restored_history
        return confirmation


    def reset(self):
        """Clear conversation history and memory state."""
        self.history       = []
        self._memory_state = "idle"
        self._session_list = []
        print("[agent] Conversation history cleared.")


# ── CLI for quick testing ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    print("\n" + "="*60)
    print("  Hardware Test Codebase Expert Agent")
    print("  Commands:")
    print("    'save as <name>'  → save current conversation")
    print("    'load'            → list and load a saved session")
    print("    'reset'           → clear conversation history")
    print("    'exit'            → quit")
    print("="*60 + "\n")

    agent = CodebaseAgent()
    print("\n\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[agent] Exiting.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("[agent] Goodbye!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            continue

        response = agent.chat(user_input)
        print(f"\nAgent: {response}\n")