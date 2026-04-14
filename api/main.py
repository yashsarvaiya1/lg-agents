from fastapi import FastAPI
from api.routes import router as invoke_router
from api.stream import router as stream_router

app = FastAPI(title="LangGraph Agent API")
app.include_router(invoke_router)
app.include_router(stream_router)
