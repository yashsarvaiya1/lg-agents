import uuid
from fastapi import APIRouter
from pydantic import BaseModel
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from agents.simple import simple_graph
from agents.serial import serial_graph
from agents.conditional import conditional_graph
from agents.loop import loop_graph
from agents.bot import bot_graph
from agents.react import react_graph
from agents.humanloopnode import humanloop_graph
from agents.humanlooptool import humanlooptool_graph
from agents.drafter import drafter_graph
from agents.subagent import subagent_graph
from agents.supervisor import supervisor_graph

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

class BotPayload(BaseModel):
    message: str
    thread_id: str = ""

class ReactPayload(BaseModel):
    message: str
    thread_id: str = ""

class HumanLoopPayload(BaseModel):
    thread_id: str

class ResumePayload(BaseModel):
    thread_id: str
    human_input: str

class HumanLoopToolPayload(BaseModel):
    message: str
    thread_id: str = ""

class DrafterPayload(BaseModel):
    message: str
    thread_id: str = ""

class SubAgentPayload(BaseModel):
    message: str
    thread_id: str = ""

class SupervisorPayload(BaseModel):
    message: str
    thread_id: str = ""
    
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

@router.post("/bot")
def run_bot(payload: BotPayload):
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = bot_graph.invoke({"messages":[{"role":"user","content":payload.message}]},config=config)
    return {"thread_id":thread_id,"response": result["messages"][-1].content}

@router.post("/react")
def run_react(payload: ReactPayload):
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = react_graph.invoke({"messages": [{"role": "user", "content": payload.message}]},config=config)
    return {"thread_id": thread_id, "response": result["messages"][-1].content}

@router.post("/humanloop/start")
def start_humanloop(payload: HumanLoopPayload):
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = humanloop_graph.invoke({"messages": []}, config=config)
    if result.get("__interrupt__"):
        return {
            "thread_id": thread_id,
            "status": "interrupted",
            "question": result["__interrupt__"][0].value
        }
    return {"thread_id": thread_id, "status": "done", "greeting": result["greeting"]}

@router.post("/humanloop/resume")
def resume_humanloop(payload: ResumePayload):
    config = {"configurable": {"thread_id": payload.thread_id}}
    result = humanloop_graph.invoke(
        Command(resume=payload.human_input),
        config=config
    )
    return {"greeting": result["greeting"], "name": result["name"]}

@router.post("/humanlooptool/start")
def start_humanlooptool(payload: HumanLoopToolPayload):
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = humanlooptool_graph.invoke(
        {"messages": [{"role": "user", "content": payload.message}], "name": None, "age": None, "greeting": None},
        config=config
    )
    if result.get("__interrupt__"):
        return {
            "thread_id": thread_id,
            "status": "interrupted",
            "question": result["__interrupt__"][0].value["question"],
            "messages": result["messages"]
        }
    return {"thread_id": thread_id, "status": "done", "greeting": result["greeting"],"messages": result["messages"]}

@router.post("/humanlooptool/resume")
def resume_humanlooptool(payload: ResumePayload):
    config = {"configurable": {"thread_id": payload.thread_id}}
    result = humanlooptool_graph.invoke(Command(resume=payload.human_input), config=config)
    if result.get("__interrupt__"):
        return {
            "thread_id": payload.thread_id,
            "status": "interrupted",
            "question": result["__interrupt__"][0].value["question"]
        }
    return {"thread_id": payload.thread_id, "status": "done", "greeting": result["greeting"]}

@router.post("/drafter/start")
def start_drafter(payload: DrafterPayload):
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = drafter_graph.invoke(
        {"messages": [{"role": "user", "content": payload.message}]},
        config=config
    )
    if result.get("__interrupt__"):
        data = result["__interrupt__"][0].value
        return {"thread_id": thread_id, "status": "interrupted", "draft": data["draft"], "question": data["message"]}
    return {"thread_id": thread_id, "status": "done", "draft": result["draft"]}


@router.post("/drafter/resume")
def resume_drafter(payload: ResumePayload):
    config = {"configurable": {"thread_id": payload.thread_id}}
    result = drafter_graph.invoke(Command(resume=payload.human_input), config=config)
    if result.get("__interrupt__"):
        data = result["__interrupt__"][0].value
        return {"thread_id": payload.thread_id, "status": "interrupted", "draft": data["draft"], "question": data["message"]}
    return {"thread_id": payload.thread_id, "status": "done", "draft": result["draft"]}

@router.post("/subagent")
def run_subagent(payload: SubAgentPayload):
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = subagent_graph.invoke(
        {"messages": [{"role": "user", "content": payload.message}]},
        config=config
    )
    return {"thread_id": thread_id, "response": result["messages"]}


@router.post("/supervisor")
def run_supervisor(payload: SupervisorPayload):
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    supervisor_graph.invoke(
        {"messages": [{"role": "user", "content": payload.message}]},
        config=config
    )
    # imp use this it have the value of merged state after the invoke, invoke only returns updated staes
    state = supervisor_graph.get_state(config)
    return {
        "thread_id": thread_id,
        "response": state.values["messages"][-1].content,
        "last_agent": state.values.get("last_agent"),
        "last_result": state.values.get("last_result"),
        "task_count": state.values.get("task_count")
    }
