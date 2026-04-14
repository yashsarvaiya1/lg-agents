from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from shared.llm import chat_llm
from shared.memory import checkpointer
from typing import Annotated


# ── Sub-agent: calculator ─────────────────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    "Evaluate a math expression like '12 * 4 + 2'."
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

calc_tools = [calculate]
calc_llm = chat_llm.bind_tools(calc_tools)

def calc_node(state: MessagesState) -> dict:
    system = SystemMessage(content="You are a calculator. Use calculate tool for all math.")
    return {"messages": [calc_llm.invoke([system] + list(state["messages"]))]}

calc_graph = StateGraph(MessagesState)
calc_graph.add_node("calc", calc_node)
calc_graph.add_node("tools", ToolNode(calc_tools))
calc_graph.add_edge(START, "calc")
calc_graph.add_conditional_edges("calc", tools_condition)
calc_graph.add_edge("tools", "calc")
calc_subgraph = calc_graph.compile()  # no checkpointer — parent owns state


# ── Parent agent ──────────────────────────────────────────────────────────────

@tool
def calculator_tool(
    expression: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig
) -> Command:
    "Delegate math calculations to the calculator sub-agent."
    result = calc_subgraph.invoke({"messages": [HumanMessage(content=expression)]})
    content = result["messages"][-1].content
    return Command(update={
        "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
    })

parent_tools = [calculator_tool]
parent_llm = chat_llm.bind_tools(parent_tools)

def llm_node(state: MessagesState) -> dict:
    system = SystemMessage(content=(
        "You are a helpful assistant. "
        "For any math or calculation tasks use calculator_tool."
    ))
    return {"messages": [parent_llm.invoke([system] + list(state["messages"]))]}

graph = StateGraph(MessagesState)
graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode(parent_tools))
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", tools_condition)
graph.add_edge("tools", "llm")
subagent_graph = graph.compile(checkpointer=checkpointer)
