from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import SystemMessage
from shared.llm import chat_llm
from shared.memory import checkpointer

SYSTEM = SystemMessage(content=(
    "You are a helpful assistant. "
    "You have full access to the conversation history above. "
    "Always refer to it when answering questions about past messages."
))

def llm_node(state: MessagesState) -> dict:
    response = chat_llm.invoke([SYSTEM] + state["messages"])
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("llm", llm_node)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)
bot_graph = graph.compile(checkpointer=checkpointer)
