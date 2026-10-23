"""
TriageAI - Multilingual Support Ticket Intelligence System
Chains together Hugging Face pipelines to process incoming tickets.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(title="TriageAI")



translator_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
translator_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment = pipeline("sentiment-analysis")
summarizer_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
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


@app.post("/process_ticket", response_model=TicketResult)
def process_ticket(ticket: Ticket):
    # 1. Translate to English (skip if already English)
    if ticket.language != "en":
        inputs = translator_tokenizer(ticket.text, return_tensors="pt", truncation=True)
        translated_ids = translator_model.generate(**inputs, max_length=100)
        translated = translator_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    else:
        translated = ticket.text

    # 2. Classify intent (zero-shot — no training needed)
    intent_result = classifier(translated, CATEGORIES)
    top_category = intent_result["labels"][0]
    top_score = intent_result["scores"][0]

    # 3. Sentiment / urgency signal
    sent_result = sentiment(translated)[0]
    urgency = "high" if sent_result["label"] == "NEGATIVE" and sent_result["score"] > 0.85 else "normal"

    # 4. Summarize (skip if text is already short)
    if len(translated.split()) > 20:
        summary_inputs = summarizer_tokenizer(translated, return_tensors="pt", truncation=True)
        summary_ids = summarizer_model.generate(**summary_inputs, max_length=30, min_length=5)
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    else:
        summary = translated

    return TicketResult(
        original_text=ticket.text,
        translated_text=translated,
        category=top_category,
        category_confidence=round(top_score, 3),
        urgency=urgency,
        sentiment=sent_result["label"],
        summary=summary,
    )


@app.get("/")
def health_check():
    return {"status": "TriageAI is running"}
