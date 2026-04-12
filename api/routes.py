import uuid
from fastapi import APIRouter
from pydantic import BaseModel
from agents.simple import simple_graph
from agents.serial import serial_graph
from agents.conditional import conditional_graph
from agents.loop import loop_graph

router = APIRouter(prefix="/agents")

class SimplePayload(BaseModel):
    message: str

class SerialPayload(BaseModel):
    name: str
    thread_id: str

class ConditionalPayload(BaseModel):
    num1: int
    num2: int
    operation: str

class LoopPayload(BaseModel):
    limit: int

@router.post("/simple")
def run_simple(payload: SimplePayload):
    result = simple_graph.invoke({"messages": [{"role":"user","content":payload.message}]})
    return {"response": result["messages"][-1].content}

@router.post("/serial")
def run_serial(payload: SerialPayload):
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id":thread_id}}    
    result = serial_graph.invoke({"name":payload.name,"greeting":""},config=config)
    return {"name":result["name"],"greeting":result["greeting"],"messages":result["messages"]}

@router.post("/conditional")
def run_conditional(payload: ConditionalPayload):
    result = conditional_graph.invoke({
        "num1": payload.num1,
        "num2": payload.num2,
        "operation": payload.operation
    })
    return {"result": result["result"]}

@router.post("/loop")
def run_loop(payload: LoopPayload):
    result = loop_graph.invoke({"counter":0,"limit":payload.limit,"items":[]})
    return {"counter":result["counter"],"limit":result["limit"],"items":result["items"]}
