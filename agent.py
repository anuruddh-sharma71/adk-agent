import os
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))

model = genai.GenerativeModel("gemini-2.0-flash")

async def run_agent(user_input: str, session_id: str = "default") -> str:
    try:
        prompt = f"""You are a text summarization assistant.
Summarize the following text clearly and concisely.
Start your response with **Summary:** followed by the summary.
Then on a new line write **Original word count:** with the word count.

Text to summarize:
{user_input}"""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"
