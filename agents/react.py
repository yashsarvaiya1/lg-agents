from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from shared.llm import chat_llm
from shared.memory import checkpointer

SYSTEM = SystemMessage(content=(
    "You are a calculator assistant. "
    "You MUST use the provided tools for ALL arithmetic. "
    "Never calculate math yourself."
))

@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return f"{a} + {b} = {a + b}"

@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers."""
    return f"{a} * {b} = {a * b}"

@tool
def subtract(a: int, b: int) -> str:
    """Subtract two numbers."""
    return f"{a} - {b} = {a - b}"

tools = [add, multiply, subtract]
llm_with_tools = chat_llm.bind_tools(tools)

def llm_node(state: MessagesState) -> dict:
    response = llm_with_tools.invoke([SYSTEM] + state["messages"])
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", tools_condition)
graph.add_edge("tools", "llm")
react_graph = graph.compile(checkpointer=checkpointer)
