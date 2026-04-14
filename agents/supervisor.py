from typing import Annotated, Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from shared.llm import chat_llm
from shared.memory import checkpointer


class State(MessagesState):
    last_agent: Optional[str] = None      # which sub-agent was last called
    last_result: Optional[str] = None     # what it returned
    task_count: int = 0                   # how many tasks routed so far


# ── Sub-agent: Drafter ────────────────────────────────────────────────────────

def drafter_node(state: MessagesState) -> dict:
    system = SystemMessage(content="You are an expert document drafter.")
    return {"messages": [chat_llm.invoke([system] + list(state["messages"]))]}

drafter_graph = StateGraph(MessagesState)
drafter_graph.add_node("drafter", drafter_node)
drafter_graph.add_edge(START, "drafter")
drafter_graph.add_edge("drafter", END)
drafter_graph = drafter_graph.compile()


# ── Sub-agent: Researcher ─────────────────────────────────────────────────────

def researcher_node(state: MessagesState) -> dict:
    system = SystemMessage(content="You are an expert researcher. Provide detailed, factual answers.")
    return {"messages": [chat_llm.invoke([system] + list(state["messages"]))]}

researcher_graph = StateGraph(MessagesState)
researcher_graph.add_node("researcher", researcher_node)
researcher_graph.add_edge(START, "researcher")
researcher_graph.add_edge("researcher", END)
researcher_graph = researcher_graph.compile()


# ── Supervisor tools ──────────────────────────────────────────────────────────

@tool
def transfer_to_drafter(
    task: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    "Transfer to the drafter agent for any writing, drafting, or document creation."
    result = drafter_graph.invoke({"messages": [HumanMessage(content=task)]})
    content = result["messages"][-1].content
    return Command(update={
        "last_agent": "drafter",
        "last_result": content,
        "task_count": 1,            # add reducer handles incrementing
        "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
    })


@tool
def transfer_to_researcher(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    "Transfer to the researcher agent for any research, analysis, or factual questions."
    result = researcher_graph.invoke({"messages": [HumanMessage(content=query)]})
    content = result["messages"][-1].content
    return Command(update={
        "last_agent": "researcher",
        "last_result": content,
        "task_count": 1,
        "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
    })


# ── Supervisor ────────────────────────────────────────────────────────────────

supervisor_tools = [transfer_to_drafter, transfer_to_researcher]
supervisor_llm = chat_llm.bind_tools(supervisor_tools)

def supervisor_node(state: State) -> dict:
    system = SystemMessage(content=(
        "You are a supervisor. Route tasks to the right specialist:\n"
        "- Use transfer_to_drafter for writing, drafting, or document creation\n"
        "- Use transfer_to_researcher for research, analysis, or factual questions\n"
        "- Answer simple questions directly without tools"
    ))
    return {"messages": [supervisor_llm.invoke([system] + list(state["messages"]))]}

graph = StateGraph(State)
graph.add_node("supervisor", supervisor_node)
graph.add_node("tools", ToolNode(supervisor_tools))
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", tools_condition)
graph.add_edge("tools", "supervisor")
supervisor_graph = graph.compile(checkpointer=checkpointer)
