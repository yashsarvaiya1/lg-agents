from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Langgraph Agents API")
app.include_router(router)
