import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from agent import run_agent

app = FastAPI(title="ADK Text Summarizer Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class RunRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None

class RunResponse(BaseModel):
    response: str
    session_id: str
    agent: str = "text_summarizer"

@app.get("/health")
async def health():
    return {"status": "ok", "agent": "text_summarizer", "model": "gemini-2.0-flash"}

@app.post("/run", response_model=RunResponse)
async def run(body: RunRequest):
    session_id = body.session_id or str(uuid.uuid4())
    try:
        result = await run_agent(body.message, session_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RunResponse(response=result, session_id=session_id)

app.add_api_route("/summarize", run, methods=["POST"], response_model=RunResponse)

# Serve frontend — must be AFTER all API routes
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"status": "ok", "agent": "text_summarizer", "model": "gemini-2.0-flash"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
