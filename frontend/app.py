"""
TriageAI Dashboard - simple UI for agents to submit and view triaged tickets.
Run with: streamlit run app.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/process_ticket"

st.set_page_config(page_title="TriageAI Dashboard", layout="centered")
st.title("🎫 TriageAI — Ticket Triage")

st.write("Paste a customer message below (any language) and let AI triage it.")

text = st.text_area("Ticket text", height=150, placeholder="e.g. Мне нужно вернуть деньги, оплата прошла дважды!")
lang = st.selectbox("Source language", ["auto", "en", "ru", "de", "kk"])

if st.button("Process Ticket"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Running AI pipeline..."):
            response = requests.post(API_URL, json={"text": text, "language": lang})
            if response.status_code == 200:
                result = response.json()
                st.success("Ticket processed!")

                col1, col2 = st.columns(2)
                col1.metric("Category", result["category"])
                col2.metric("Urgency", result["urgency"])

                st.write("**Sentiment:**", result["sentiment"])
                st.write("**Summary:**", result["summary"])
                st.write("**Translated text:**", result["translated_text"])
            else:
                st.error(f"Error: {response.text}")
