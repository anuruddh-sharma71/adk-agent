import os
import uuid
import json
import urllib.request
import urllib.error
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="SummarAI Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class RunRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    groq_key: str | None = None

class RunResponse(BaseModel):
    response: str
    session_id: str
    agent: str = "summarai"

@app.post("/run", response_model=RunResponse)
async def run(body: RunRequest):
    session_id = body.session_id or str(uuid.uuid4())
    groq_key = body.groq_key or os.environ.get("GROQ_API_KEY", "")

    if not groq_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY not set")

    try:
        payload = json.dumps({
            "model": "llama3-8b-8192",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are SummarAI, a powerful and friendly AI assistant. "
                        "You can answer any question, summarize text, explain concepts, "
                        "write content, help with code, give advice, and much more. "
                        "When asked to summarize, start with **Summary:** and end with **Word count:** X words. "
                        "For all other questions, give clear, helpful, well-formatted answers. "
                        "Use **bold** for important terms. Be concise but complete."
                    )
                },
                {"role": "user", "content": body.message}
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as res:
            data = json.loads(res.read().decode("utf-8"))
            return RunResponse(
                response=data["choices"][0]["message"]["content"],
                session_id=session_id
            )

    except urllib.error.HTTPError as e:
        body_err = e.read().decode("utf-8")
        raise HTTPException(status_code=e.code, detail=f"Groq API error: {body_err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "agent": "summarai", "model": "llama3-8b-8192", "provider": "groq"}

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
