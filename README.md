# TriageAI — Multilingual Support Ticket Intelligence
🔗 **Live demo:** https://triageai-2026.streamlit.app

Automatically translates, classifies, analyzes sentiment and summarizes support tickets using an LLM via the Groq API.

## Project structure
```
triageai/
├── backend/
│   ├── main.py            # FastAPI app, sends tickets to the Groq LLM API
│   └── requirements.txt   # backend dependencies
├── frontend/
│   ├── app.py             # Streamlit dashboard
│   └── requirements.txt   # frontend dependencies
├── .gitignore
└── README.md
```

