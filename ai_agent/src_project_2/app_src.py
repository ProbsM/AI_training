"""
app.py
------
Chainlit chat UI for the Hardware Test Codebase Expert Agent.

Wraps src_agent.py with a browser-based chat interface.
Supports markdown, syntax-highlighted code blocks, and streaming.

Usage:
    chainlit run app_src.py
"""

import chainlit as cl
from src_agent import CodebaseAgent


# ── Agent Lifecycle ───────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """
    Called once when a user opens the chat.
    Initializes a fresh CodebaseAgent and stores it in the session
    so it persists across messages.
    """
    agent = CodebaseAgent()
    cl.user_session.set("agent", agent)

    await cl.Message(
        content=(
            "👋 **SRC Codebase Agent** ready.\n\n"
            "I have deep knowledge of your codebase and can help you:\n"
            "- **Answer questions** about how the code works\n"
            "- **Debug issues** — paste your log output and I'll diagnose it\n"
            "- **Generate new test scripts** modeled on existing code\n"
            "**Memory commands:**\n"
            "- `save as <name>` — save this conversation\n"
            "- `load` — load a previous conversation\n"
            "- `reset` — clear conversation history\n\n"
            "What would you like to know?"
        )
    ).send()


# ── Message Handler ───────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    """
    Called on every user message.
    Retrieves the agent from session, calls agent.chat(), and streams the response.
    """
    agent: CodebaseAgent = cl.user_session.get("agent")

    # Show a thinking indicator while the agent is working
    async with cl.Step(name="Searching codebase...") as step:
        step.output = "Retrieving relevant code chunks and generating response..."

    # Call the agent (this does RAG retrieval + Claude call)
    response = await cl.make_async(agent.chat)(message.content)

    # Send the response as a new message
    await cl.Message(content=response).send()


# ── Session End ───────────────────────────────────────────────────────────────

@cl.on_chat_end
async def on_chat_end():
    """Called when the user closes the chat. Clean up the agent."""
    agent: CodebaseAgent = cl.user_session.get("agent")
    if agent:
        agent.reset()