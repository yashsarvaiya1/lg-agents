import uuid
from fastapi import APIRouter
from pydantic import BaseModel
from agents.simple import simple_graph
from agents.serial import serial_graph

router = APIRouter(prefix="/agents")

class SimplePayload(BaseModel):
    message: str

class SerialPayload(BaseModel):
    name: str
    thread_id: str

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

