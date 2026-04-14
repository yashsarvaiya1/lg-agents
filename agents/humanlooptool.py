from typing import Annotated, Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage, SystemMessage
from shared.llm import chat_llm
from shared.memory import checkpointer


class State(MessagesState):
    name: Optional[str] = None
    age: Optional[int] = None
    greeting: Optional[str] = None


SYSTEM = SystemMessage(content=(
    "You are a greeting assistant. "
    "You MUST immediately call greeting_tool with name='' and age=0. "
    "Do NOT ask the user for name or age in plain text. "
    "Always call the tool first."
))


@tool
def greeting_tool(
    name: str,
    age: int,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    "Generate a personalised greeting. Always call with name='' and age=0 initially."
    if not name:
        name = interrupt({"question": "What is your name?"})
    if not age:
        age = interrupt({"question": "What is your age?"})
    greeting = f"Hello, {name}. You are {age} years old."
    return Command(update={
        "name": name,
        "age": age,
        "greeting": greeting,
        "messages": [ToolMessage(content=greeting, tool_call_id=tool_call_id)]
    })


tools = [greeting_tool]
llm_with_tools = chat_llm.bind_tools(tools)


def llm_node(state: State) -> dict:
    return {"messages": [llm_with_tools.invoke([SYSTEM] + state["messages"])]}


graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", tools_condition)
graph.add_edge("tools", "llm")
humanlooptool_graph = graph.compile(checkpointer=checkpointer)
