import uuid
from typing import AsyncIterator, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langgraph.types import Command
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


router = APIRouter(prefix="/stream")


AGENTS = {
    "simple": simple_graph,
    "serial": serial_graph,
    "conditional": conditional_graph,
    "loop": loop_graph,
    "bot": bot_graph,
    "react": react_graph,
    "humanloop": humanloop_graph,
    "humanlooptool": humanlooptool_graph,
    "drafter": drafter_graph,
    "subagent": subagent_graph,
    "supervisor": supervisor_graph,
}


# ── Payloads ──────────────────────────────────────────────────────────────────

class StreamPayload(BaseModel):
    message: str
    thread_id: str = ""

class ResumePayload(BaseModel):
    thread_id: str
    human_input: Any


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_agent(name: str):
    if name not in AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found. Available: {list(AGENTS.keys())}")
    return AGENTS[name]


async def event_stream(graph, inputs, config) -> AsyncIterator[str]:
    async for chunk_type, data in graph.astream(inputs, config=config, stream_mode=["updates", "messages"]):

        if chunk_type == "updates":
            if "__interrupt__" in data:
                interrupt_val = data["__interrupt__"][0].value
                yield f"event: interrupt\ndata: {interrupt_val}\n\n"
            else:
                for node_name in data.keys():
                    yield f"event: update\ndata: node={node_name}\n\n"

        elif chunk_type == "messages":
            msg, metadata = data
            if hasattr(msg, "content") and msg.content:
                yield f"event: token\ndata: {msg.content}\n\n"

    yield f"event: done\ndata: thread_id={config['configurable']['thread_id']}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/{agent}")
async def stream_agent(agent: str, payload: StreamPayload):
    graph = get_agent(agent)
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [{"role": "user", "content": payload.message}]}
    return StreamingResponse(
        event_stream(graph, inputs, config),
        media_type="text/event-stream",
        headers={"X-Thread-Id": thread_id}   # ← thread_id in response header
    )


@router.post("/{agent}/resume")
async def resume_agent(agent: str, payload: ResumePayload):
    graph = get_agent(agent)
    config = {"configurable": {"thread_id": payload.thread_id}}
    return StreamingResponse(
        event_stream(graph, Command(resume=payload.human_input), config),
        media_type="text/event-stream",
        headers={"X-Thread-Id": payload.thread_id}
    )


@router.get("/{agent}/thread/{thread_id}")
async def get_thread(agent: str, thread_id: str):
    graph = get_agent(agent)
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    return {
        "thread_id": thread_id,
        "status": "interrupted" if state.next else "done",
        "next_nodes": list(state.next),
        "state": state.values
    }


@router.delete("/{agent}/thread/{thread_id}")
async def delete_thread(agent: str, thread_id: str):
    graph = get_agent(agent)
    config = {"configurable": {"thread_id": thread_id}}
    await graph.aget_state(config)
    return {"deleted": thread_id}
