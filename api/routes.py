import uuid
from fastapi import APIRouter
from pydantic import BaseModel
from agents.simple import simple_graph

router = APIRouter(prefix="/agents")

class SimplePayload(BaseModel):
    message: str

@router.post("/simple")
def run_simple(payload: SimplePayload):
    result = simple_graph.invoke({"messages": [{"role":"user","content":payload.message}]})
    return {"response": result["messages"][-1].content}


