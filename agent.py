import os
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

def summarize_text(text: str, max_sentences: int = 3) -> dict:
    """Summarize the provided text."""
    word_count = len(text.split())
    return {"original_word_count": word_count, "max_sentences": max_sentences, "text": text}

summarize_tool = FunctionTool(func=summarize_text)

def create_agent() -> Agent:
    return Agent(
        name="text_summarizer",
        model="gemini-2.0-flash",
        description="Summarizes any text into clear, concise sentences.",
        instruction=(
            "You are a text summarization assistant. "
            "When the user provides text, call the summarize_text tool first to get metadata, "
            "then produce a clear, concise summary in the requested number of sentences. "
            "Always start your response with '**Summary:**' followed by the summary, "
            "then on a new line add '**Original word count:** X words'."
        ),
        tools=[summarize_tool],
    )

_session_service = InMemorySessionService()
_runner = None

def get_runner() -> Runner:
    global _runner
    if _runner is None:
        _runner = Runner(agent=create_agent(), app_name="summarizer_app", session_service=_session_service)
    return _runner

async def run_agent(user_input: str, session_id: str = "default") -> str:
    runner = get_runner()
    existing = await _session_service.get_session(app_name="summarizer_app", user_id="user", session_id=session_id)
    if existing is None:
        await _session_service.create_session(app_name="summarizer_app", user_id="user", session_id=session_id)
    message = types.Content(role="user", parts=[types.Part(text=user_input)])
    final_response = ""
    async for event in runner.run_async(user_id="user", session_id=session_id, new_message=message):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text
            break
    return final_response or "No response generated."
