from langgraph.graph import StateGraph,START,END,MessagesState
from shared.memory import checkpointer

class State(MessagesState):
    name: str
    greeting: str

def build_greeting(state: State):
    return {"greeting":f"Good Morning {state["name"]}.","messages":[{"role":"user","content":"simple message"}]}


graph = StateGraph(State)
graph.add_node("build_greeting", build_greeting)
graph.add_edge(START, "build_greeting")
graph.add_edge("build_greeting", END)
serial_graph = graph.compile(checkpointer=checkpointer)
