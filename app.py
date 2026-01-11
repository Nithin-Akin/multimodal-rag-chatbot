import streamlit as st
import requests
import os
import subprocess

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Multi-Modal Document Intelligence",
    layout="wide",
    page_icon="📄"
)

# -------------------------
# Styling
# -------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
}

.chat-bubble-user {
    background: #1f2937;
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 10px;
    color: #60a5fa;
}

.chat-bubble-bot {
    background: #111827;
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 6px;
    color: #e5e7eb;
}

.citation {
    font-size: 12px;
    color: #9ca3af;
    margin-bottom: 20px;
}

.sidebar-box {
    background: #111827;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
}

.big-title {
    font-size: 32px;
    font-weight: 700;
    color: #e5e7eb;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<div class='big-title'>🧠 Multi-Modal Document Intelligence</div>", unsafe_allow_html=True)
st.markdown("Ask complex questions across **text, tables, and scanned documents** using an AI-powered RAG system.")


with st.sidebar:
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.subheader("📤 Upload PDF")

    uploaded = st.file_uploader("Upload IMF / Policy PDF", type=["pdf"])

    if uploaded:
        os.makedirs("uploads", exist_ok=True)
        with open(f"uploads/{uploaded.name}", "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("PDF uploaded")

        if st.button("⚡ Ingest Document"):
            with st.spinner("Parsing text, tables & images…"):
                subprocess.run(["python", "multimodal_ingest.py"])
            st.success("Document indexed")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("—")
    st.markdown("**What this supports:**")
    st.markdown("• Text paragraphs\n• Financial tables\n• Scanned images (OCR)\n• Citations")

#chat
st.markdown("### Ask Seeing-Through-Documents AI")

if "chat" not in st.session_state:
    st.session_state.chat = []

question = st.text_input("Type a question about the document")

if st.button("Ask") and question:
    with st.spinner("Searching documents…"):
        r = requests.post(
            f"{BACKEND_URL}/ask",
            json={"question": question}
        ).json()

    st.session_state.chat.append({
        "q": question,
        "a": r["answer"],
        "c": r["citations"]
    })

#showing history
for msg in st.session_state.chat[::-1]:
    st.markdown(f"<div class='chat-bubble-user'>🧑 {msg['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble-bot'>🤖 {msg['a']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='citation'>Sources: {msg['c']}</div>", unsafe_allow_html=True)