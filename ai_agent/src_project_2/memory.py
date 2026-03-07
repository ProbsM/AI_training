"""
memory.py
---------
Session memory management for the Hardware Test Codebase Agent.

Handles saving and loading conversation sessions to/from JSON files.
When loading, summarizes the full history into a compact summary using
Claude, then prepends it to the restored history so context is preserved
without blowing up the context window.

This module is not called directly — it is used by agent.py when the
user asks to save or load a conversation.
"""

import os
import json
from pathlib import Path
from datetime import datetime

from anthropic import Anthropic


# ── Configuration ─────────────────────────────────────────────────────────────

SESSIONS_DIR    = "."                        # Same folder as agent.py
SESSION_EXT     = ".json"
SUMMARY_MODEL   = "claude-haiku-4-5-20251001"
MAX_HISTORY     = 20                         # Max turns to keep after loading


# ── Save ──────────────────────────────────────────────────────────────────────

def save_session(history: list[dict], session_name: str) -> str:
    """
    Save the full conversation history to a JSON file.
    Saves ALL turns regardless of MAX_HISTORY — no information is lost on disk.

    Returns the path of the saved file.
    """
    # Sanitize session name — remove special characters, replace spaces with underscores
    safe_name = "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in session_name.strip()
    ).strip("_")

    if not safe_name:
        safe_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    file_path = Path(SESSIONS_DIR) / f"{safe_name}{SESSION_EXT}"

    session_data = {
        "session_name": safe_name,
        "saved_at":     datetime.now().isoformat(),
        "turn_count":   len(history) // 2,
        "history":      history,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    return str(file_path)


# ── List ──────────────────────────────────────────────────────────────────────

def list_sessions() -> list[dict]:
    """
    List all saved session files in the sessions directory.

    Returns a list of dicts with session metadata, sorted by most recent first:
      [{"index": 1, "name": "voltage_debug", "saved_at": "...", "turns": 12}, ...]
    """
    sessions_path = Path(SESSIONS_DIR)
    session_files = sorted(
        sessions_path.glob(f"*{SESSION_EXT}"),
        key=lambda f: f.stat().st_mtime,
        reverse=True  # Most recent first
    )

    sessions = []
    for i, file_path in enumerate(session_files, 1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "index":    i,
                "name":     data.get("session_name", file_path.stem),
                "saved_at": data.get("saved_at", "unknown"),
                "turns":    data.get("turn_count", 0),
                "path":     str(file_path),
            })
        except Exception:
            # Skip malformed session files
            continue

    return sessions


def format_session_list(sessions: list[dict]) -> str:
    """Format the session list into a readable string for the agent to display."""
    if not sessions:
        return "No saved sessions found."

    lines = ["Here are your saved sessions:\n"]
    for s in sessions:
        # Parse and reformat the saved_at timestamp
        try:
            dt = datetime.fromisoformat(s["saved_at"])
            saved_str = dt.strftime("%b %d %Y at %I:%M %p")
        except Exception:
            saved_str = s["saved_at"]

        lines.append(f"  {s['index']}. {s['name']}  ({s['turns']} turns — saved {saved_str})")

    lines.append("\nWhich session would you like to load? (enter the number)")
    return "\n".join(lines)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_session(session_index: int, sessions: list[dict], anthropic_client: Anthropic) -> tuple[list[dict], str]:
    """
    Load a saved session by its index in the sessions list.

    Steps:
      1. Read full history from JSON file
      2. If history exceeds MAX_HISTORY turns, summarize it with Claude
      3. Prepend the summary as a system-style message
      4. Append the most recent MAX_HISTORY turns after the summary
      5. Return the restored history + a confirmation message for the agent

    Returns:
      (restored_history, confirmation_message)
    """
    # Validate index
    if session_index < 1 or session_index > len(sessions):
        return None, f"Invalid selection. Please enter a number between 1 and {len(sessions)}."

    session_meta = sessions[session_index - 1]

    try:
        with open(session_meta["path"], "r", encoding="utf-8") as f:
            session_data = json.load(f)
    except Exception as e:
        return None, f"Failed to load session '{session_meta['name']}': {e}"

    full_history = session_data.get("history", [])
    total_turns  = len(full_history) // 2

    # ── If history is short enough, restore as-is ─────────────────────────
    if total_turns <= MAX_HISTORY:
        confirmation = (
            f"Loaded session **{session_meta['name']}** "
            f"({total_turns} turns). Conversation restored — you can pick up where you left off."
        )
        return full_history, confirmation

    # ── If history is long, summarize then restore recent turns ───────────
    summary = _summarize_history(full_history, anthropic_client, session_meta["name"])

    # Keep only the most recent MAX_HISTORY turns
    recent_history = full_history[-(MAX_HISTORY * 2):]

    # Prepend the summary as a user+assistant exchange so it fits the
    # messages format Claude expects
    summary_exchange = [
        {
            "role": "user",
            "content": "Please summarize what we have discussed so far in this session."
        },
        {
            "role": "assistant",
            "content": f"Here is a summary of our previous conversation:\n\n{summary}"
        }
    ]

    restored_history = summary_exchange + recent_history

    confirmation = (
        f"Loaded session **{session_meta['name']}** "
        f"({total_turns} turns total). The full session was summarized and the most recent "
        f"{MAX_HISTORY} turns were restored. Here's what we covered:\n\n{summary}"
    )

    return restored_history, confirmation


def _summarize_history(history: list[dict], client: Anthropic, session_name: str) -> str:
    """
    Use Claude Haiku to summarize a full conversation history into a
    compact plain-English summary.

    Strips the injected RAG context from user messages before summarizing
    to keep the summary focused on the actual conversation content.
    """
    # Extract just the actual user questions (strip the RAG context blocks)
    cleaned_turns = []
    for msg in history:
        if msg["role"] == "user":
            content = msg["content"]
            # Strip the retrieved code context block we inject in agent.py
            if "## User Question" in content:
                content = content.split("## User Question")[-1].strip()
            cleaned_turns.append(f"User: {content}")
        elif msg["role"] == "assistant":
            cleaned_turns.append(f"Assistant: {msg['content']}")

    conversation_text = "\n\n".join(cleaned_turns)

    prompt = f"""The following is a conversation between an engineer and an AI assistant that is an expert on a hardware instrument testing codebase.

Summarize this conversation in 5-10 sentences. Focus on:
- What specific problems or questions the engineer was working on
- Which files, functions, or modules were discussed
- What solutions, fixes, or new code was produced
- Any important conclusions or decisions made

Be specific — include actual function names, file names, and technical details where relevant.

Conversation:
{conversation_text}

Summary:"""

    response = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()
