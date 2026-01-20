import os
from dotenv import load_dotenv
load_dotenv()

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except Exception:
    _GROQ_AVAILABLE = False

GROQ_API_KEY = os.getenv('GROQ_API_KEY')


def _format_history(history):
    messages = []
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    return messages


def generate_answer_with_groq(context: str, query: str, history=None, max_tokens: int = 800):
    if not _GROQ_AVAILABLE:
        return "[Groq client not available — install groq and set GROQ_API_KEY]"
    client = Groq(api_key=GROQ_API_KEY)

    messages = []
    if history:
        messages.extend(_format_history(history))

    messages.append({
        "role": "system",
        "content": (
            "You are an expert medical research assistant. Answer using only the provided context. "
            "Cite sources by filename and chunk identifiers where appropriate. Be concise and factual."
        )
    })

    user_content = f"Context:\n{context}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_content})

    completion = client.chat.completions.create(
       model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens
    )
    # The response object shape follows Groq's python client; adjust if fields differ
    return completion.choices[0].message.content
