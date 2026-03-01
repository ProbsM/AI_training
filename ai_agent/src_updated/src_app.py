import gradio as gr
from main import run_agent, llm, llm_no_tools, chat_history_qa, chat_history_dbg, choose_agent

def handle(user_input):
    mode = choose_agent(user_input)
    history = chat_history_dbg if mode == "debug" else chat_history_qa
    answer = run_agent(
        llm=llm,
        llm_no_tools=llm_no_tools,
        chat_history=history,
        user_request=user_input,
    )
    return f"[{mode.upper()}]\n{answer}"

demo = gr.ChatInterface(fn=handle)
demo.launch()