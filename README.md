# TriageAI — Multilingual Support Ticket Intelligence

Automatically translates, classifies, scores urgency, and summarizes
incoming customer support tickets using Hugging Face pipelines.

## Project structure
triageai/
├── backend/
│   ├── main.py            # FastAPI app, sends tickets to the Groq LLM API
│   └── requirements.txt   # backend dependencies
├── frontend/
│   ├── app.py             # Streamlit dashboard
│   └── requirements.txt   # frontend dependencies
├── .gitignore
└── README.md


