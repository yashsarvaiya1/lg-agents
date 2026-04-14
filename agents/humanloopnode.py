from typing import Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import interrupt
from shared.memory import checkpointer

class State(MessagesState):
    name: Optional[str] = None
    greeting: Optional[str] = None

def greeting_node(state: State) -> dict:
    name = state.get("name") or interrupt("Please provide your name:")
    return {"name": name, "greeting": f"Hello, {name}! Warm greetings."}

graph = StateGraph(State)
graph.add_node("greeting", greeting_node)
graph.add_edge(START, "greeting")
graph.add_edge("greeting", END)
humanloop_graph = graph.compile(checkpointer=checkpointer)
