import os
from google import genai

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))

async def run_agent(user_input: str, session_id: str = "default") -> str:
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_input
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"
