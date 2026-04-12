from typing import Literal
from langgraph.graph import StateGraph, START, END, MessagesState


class State(MessagesState):
    num1: int
    num2: int
    operation: str
    result: str

def route_operation(state: State) -> Literal["sum", "sub"]:
    return "sum" if state["operation"] == "sum" else "sub"

def sum_node(state: State) -> dict:
    return {"result": f"{state['num1']} + {state['num2']} = {state['num1'] + state['num2']}"}

def sub_node(state: State) -> dict:
    return {"result": f"{state['num1']} - {state['num2']} = {state['num1'] - state['num2']}"}

graph = StateGraph(State)
graph.add_node("sum", sum_node)
graph.add_node("sub", sub_node)
graph.add_conditional_edges(START, route_operation, {"sum": "sum", "sub": "sub"})
graph.add_edge("sum", END)
graph.add_edge("sub", END)
conditional_graph = graph.compile()
