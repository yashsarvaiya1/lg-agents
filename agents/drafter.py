from typing import Annotated, Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage, SystemMessage
from shared.llm import chat_llm
from shared.memory import checkpointer


class State(MessagesState):
    draft: Optional[str] = None
    feedback: Optional[str] = None
    satisfied: Optional[bool] = None
    counter: int = 0


@tool
def extract_satisfaction(
    draft: str,
    counter: int,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    "Show the draft to the user and collect satisfaction feedback via interrupt."
    feedback = interrupt({"message": "Are you satisfied? (yes / provide feedback)", "draft": draft})
    satisfied = feedback.strip().lower() == "yes"
    return Command(update={
        "draft": draft,
        "feedback": feedback,
        "counter": counter + 1,
        "satisfied": satisfied,
        "messages": [ToolMessage(content=f"User feedback: {feedback}", tool_call_id=tool_call_id)]
    })


tools = [extract_satisfaction]
llm_drafting = chat_llm.bind_tools(tools)


def drafting_node(state: State) -> dict:
    satisfied = state.get("satisfied")
    counter = state.get("counter") or 0

    if satisfied:
        return {"messages": [chat_llm.invoke(state["messages"])]}

    if counter == 0:
        system = SystemMessage(content=(
            "You are a drafting assistant. When asked to draft something, "
            "create the document and immediately call extract_satisfaction "
            "with the full draft text and counter=0."
        ))
    else:
        system = SystemMessage(content=(
            f"Improve this draft based on user feedback.\n"
            f"CURRENT DRAFT: {state.get('draft')}\n"
            f"FEEDBACK: {state.get('feedback')}\n"
            f"Create the improved version and call extract_satisfaction "
            f"with the new draft and counter={counter}."
        ))

    return {"messages": [llm_drafting.invoke([system] + list(state["messages"]))]}


graph = StateGraph(State)
graph.add_node("drafter", drafting_node,
    retry=RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0))
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "drafter")
graph.add_conditional_edges("drafter", tools_condition)
graph.add_edge("tools", "drafter")
drafter_graph = graph.compile(checkpointer=checkpointer)
