from langgraph.graph import StateGraph,START,END,MessagesState

def greet(state:MessagesState):
    return {"messages": [{"role":"assistant", "content": f"Hello! you said {state['messages'][-1].content}"}]}

graph = StateGraph(MessagesState)
graph.add_node("greet",greet)
graph.add_edge(START,"greet")
graph.add_edge("greet",END)

simple_graph = graph.compile()
