"""
Streamlit chat UI — what the user sees.
Beautiful, mobile-friendly banking chat interface.
"""

import streamlit as st
import requests
import uuid

# ⚠️ Change this to your Render URL when deployed
BACKEND_URL = "https://banking-chatbot-c9fo.onrender.com"

# Page config
st.set_page_config(
    page_title="Bank Assistant",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .bank-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        color: white;
        padding: 25px 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(26, 35, 126, 0.3);
    }
    .bank-header h2 { margin: 0; font-size: 24px; }
    .bank-header p { margin: 8px 0 0 0; opacity: 0.85; font-size: 14px; }
    .source-box {
        background: #e3f2fd;
        border-left: 3px solid #1976d2;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 12px;
        color: #1565c0;
        margin-top: 5px;
    }
    .stChatMessage { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="bank-header">
    <h2>🏦 Digital Bank Assistant</h2>
    <p>Ask me anything about accounts, loans, cards, KYC, or fraud prevention</p>
</div>
""", unsafe_allow_html=True)

# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            sources_text = " · ".join(msg["sources"])
            st.markdown(f'<div class="source-box">📄 From: {sources_text}</div>', unsafe_allow_html=True)

# Show quick suggestion buttons only at the start
if len(st.session_state.messages) == 0:
    st.markdown("**💡 Common questions — click to ask:**")
    
    row1 = st.columns(2)
    row2 = st.columns(2)
    
    suggestions = [
        ("🔑", "How do I reset my ATM PIN?"),
        ("🏠", "Documents needed for home loan?"),
        ("📋", "How do I complete my KYC?"),
        ("🚨", "What to do if I suspect fraud?"),
    ]
    
    for idx, (icon, text) in enumerate(suggestions):
        col = row1[idx] if idx < 2 else row2[idx - 2]
        with col:
            if st.button(f"{icon} {text}", key=f"q{idx}", use_container_width=True):
                st.session_state.pending_question = text
                st.rerun()

# Handle pending question from buttons
question = st.session_state.pending_question
st.session_state.pending_question = None

# Text input
typed_question = st.chat_input("Type your banking question here...")
if typed_question:
    question = typed_question

# Process question
if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "question": question
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    st.write(answer)
                    
                    if sources:
                        sources_text = " · ".join(sources)
                        st.markdown(f'<div class="source-box">📄 From: {sources_text}</div>', unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                else:
                    error_msg = "Sorry, I encountered an error. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
                    
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot connect to backend. Make sure FastAPI server is running on port 8000.")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please try again.")

# Sidebar info
with st.sidebar:
    st.markdown("### 🏦 Bank Assistant")
    st.markdown("**AI Stack (All FREE):**")
    st.markdown("- 🤖 Groq Llama 3.1\n- 🔍 HuggingFace Embeddings\n- 🗄️ ChromaDB\n- ⚡ FastAPI")
    st.divider()
    st.markdown("**Topics:**")
    st.markdown("- 💳 Cards & Payments\n- 🏠 Loans\n- 📋 KYC Process\n- 🔒 Fraud Prevention\n- 💰 Accounts")
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        try:
            requests.delete(f"{BACKEND_URL}/session/{st.session_state.session_id}", timeout=5)
        except:
            pass
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.caption(f"Session: {st.session_state.session_id[:8]}...")