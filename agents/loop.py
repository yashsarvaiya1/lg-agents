from langgraph.graph import StateGraph, START, END, MessagesState

class State(MessagesState):
    counter: int
    limit: int
    items: list

def loop_node(state: State):
    updated = state["items"] + [f"Item {state["counter"]+1}."]
    return {"items": updated, "counter":state["counter"]+1}

def should_continue(state:State):
    return "loop" if state["counter"] < state["limit"] else "exit"

graph = StateGraph(State)
graph.add_node("loop",loop_node)
graph.add_edge(START,"loop")
graph.add_conditional_edges("loop",should_continue,{"loop":"loop","exit":END})
loop_graph = graph.compile()
