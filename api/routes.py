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

@router.post("/simple")
def run_simple(payload: SimplePayload):
    result = simple_graph.invoke({"messages": [{"role":"user","content":payload.message}]})
    return {"response": result["messages"][-1].content}

@router.post("/serial")
def run_serial(payload: SerialPayload):
    result = serial_graph.invoke({"name":payload.name,"greeting":""})
    return {"name":result["name"],"greeting":result["greeting"]}

