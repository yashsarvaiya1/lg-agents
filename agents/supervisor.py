from typing import Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from shared.llm import chat_llm
from shared.memory import checkpointer


# ── Sub-agent: Drafter ────────────────────────────────────────────────────────

def drafter_node(state: MessagesState) -> dict:
    system = SystemMessage(content="You are an expert document drafter.")
    return {"messages": [chat_llm.invoke([system] + list(state["messages"]))]}

drafter_graph = StateGraph(MessagesState)
drafter_graph.add_node("drafter", drafter_node)
drafter_graph.add_edge(START, "drafter")
drafter_graph.add_edge("drafter", END)
drafter_graph = drafter_graph.compile()  # no checkpointer


# ── Sub-agent: Researcher ─────────────────────────────────────────────────────

def researcher_node(state: MessagesState) -> dict:
    system = SystemMessage(content="You are an expert researcher. Provide detailed, factual answers.")
    return {"messages": [chat_llm.invoke([system] + list(state["messages"]))]}

researcher_graph = StateGraph(MessagesState)
researcher_graph.add_node("researcher", researcher_node)
researcher_graph.add_edge(START, "researcher")
researcher_graph.add_edge("researcher", END)
researcher_graph = researcher_graph.compile()  # no checkpointer


# ── Supervisor tools ──────────────────────────────────────────────────────────

@tool
def transfer_to_drafter(
    task: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    "Transfer to the drafter agent for any writing, drafting, or document creation."
    result = drafter_graph.invoke({"messages": [HumanMessage(content=task)]})
    return Command(update={
        "messages": [ToolMessage(content=result["messages"][-1].content, tool_call_id=tool_call_id)]
    })


@tool
def transfer_to_researcher(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    "Transfer to the researcher agent for any research, analysis, or factual questions."
    result = researcher_graph.invoke({"messages": [HumanMessage(content=query)]})
    return Command(update={
        "messages": [ToolMessage(content=result["messages"][-1].content, tool_call_id=tool_call_id)]
    })


# ── Supervisor ────────────────────────────────────────────────────────────────

supervisor_tools = [transfer_to_drafter, transfer_to_researcher]
supervisor_llm = chat_llm.bind_tools(supervisor_tools)

def supervisor_node(state: MessagesState) -> dict:
    system = SystemMessage(content=(
        "You are a supervisor. You MUST route tasks strictly:\n"
        "- ALWAYS use transfer_to_drafter for ANY writing or drafting task\n"
        "- ALWAYS use transfer_to_researcher for ANY research or factual question\n"
        "- Only answer directly if the question is a simple greeting or meta question"
    ))
    return {"messages": [supervisor_llm.invoke([system] + list(state["messages"]))]}

graph = StateGraph(MessagesState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("tools", ToolNode(supervisor_tools))
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", tools_condition)
graph.add_edge("tools", "supervisor")
supervisor_graph = graph.compile(checkpointer=checkpointer)
