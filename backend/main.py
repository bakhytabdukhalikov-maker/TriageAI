"""
TriageAI - Multilingual Support Ticket Intelligence System
Uses an external LLM API (Groq by default) for translation, classification,
sentiment analysis and summarization, so no models are loaded into RAM.
The /process_ticket request and response format is unchanged.
"""

import json
import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="TriageAI")

# Settings come from Render environment variables (never hard-code the key)
LLM_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

CATEGORIES = ["billing", "technical bug", "complaint", "feature request", "general question"]


class Ticket(BaseModel):
    text: str
    language: str = "auto"  # e.g. "ru", "kk", "de", "en"


class TicketResult(BaseModel):
    original_text: str
    translated_text: str
    category: str
    category_confidence: float
    urgency: str
    sentiment: str
    summary: str


SYSTEM_PROMPT = f"""You are a customer support ticket analyst.
Given a support ticket (any language), return ONLY a JSON object with these keys:
- "translated_text": the ticket translated into English (unchanged if already English)
- "category": exactly one of {json.dumps(CATEGORIES)}
- "category_confidence": a number from 0 to 1 for how sure you are of the category
- "sentiment": exactly "POSITIVE" or "NEGATIVE"
- "sentiment_score": a number from 0 to 1 for how strong that sentiment is
- "summary": a one-sentence English summary, at most 25 words
No extra text, no markdown."""


def call_llm(ticket_text: str, language: str) -> dict:
    if not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set on the server")

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Ticket language hint: {language}\n\nTicket:\n{ticket_text}"},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if LLM_MODEL.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"  # faster and uses fewer free-tier tokens

    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}

    for attempt in range(2):  # one retry if rate-limited
        try:
            resp = httpx.post(f"{LLM_BASE_URL}/chat/completions", json=payload,
                              headers=headers, timeout=60)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Could not reach AI provider: {e}")
        if resp.status_code == 429 and attempt == 0:
            time.sleep(3)
            continue
        break

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"AI provider error {resp.status_code}: {resp.text[:300]}")

    content = resp.json()["choices"][0]["message"]["content"]
    try:
        start, end = content.find("{"), content.rfind("}")
        return json.loads(content[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=502, detail="AI provider returned invalid JSON")


def to_float(value, default=0.5) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return default


@app.post("/process_ticket", response_model=TicketResult)
def process_ticket(ticket: Ticket):
    result = call_llm(ticket.text, ticket.language)

    # 1. Translation (keep original text if it's already English, as before)
    if ticket.language == "en":
        translated = ticket.text
    else:
        translated = str(result.get("translated_text") or ticket.text).strip()

    # 2. Category, forced into the allowed list
    category = str(result.get("category", "")).strip().lower()
    if category not in CATEGORIES:
        category = "general question"
    confidence = to_float(result.get("category_confidence"))

    # 3. Sentiment + same urgency rule as the original version
    sentiment = str(result.get("sentiment", "")).strip().upper()
    if sentiment not in ("POSITIVE", "NEGATIVE"):
        sentiment = "NEGATIVE" if category == "complaint" else "POSITIVE"
    sentiment_score = to_float(result.get("sentiment_score"))
    urgency = "high" if sentiment == "NEGATIVE" and sentiment_score > 0.85 else "normal"

    # 4. Summary (short texts are returned as-is, as before)
    if len(translated.split()) > 20:
        summary = str(result.get("summary") or translated).strip()
    else:
        summary = translated

    return TicketResult(
        original_text=ticket.text,
        translated_text=translated,
        category=category,
        category_confidence=round(confidence, 3),
        urgency=urgency,
        sentiment=sentiment,
        summary=summary,
    )


@app.get("/")
def health_check():
    return {"status": "TriageAI is running"}
