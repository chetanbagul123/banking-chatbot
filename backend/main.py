"""
FastAPI server — the backend.
Receives questions via HTTP, returns AI answers.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag_pipeline import create_qa_chain, get_answer
import uvicorn

app = FastAPI(
    title="🏦 Banking Chatbot API",
    description="AI customer service powered by Groq + LangChain + ChromaDB",
    version="1.0.0"
)

# Allow frontend (Streamlit) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Store chat chains per session (each user gets their own memory)
session_chains = {}

# Request and Response shapes
class ChatRequest(BaseModel):
    session_id: str    # Unique ID per user (frontend generates this)
    question: str      # What the user asked

class ChatResponse(BaseModel):
    answer: str
    sources: list
    session_id: str

@app.get("/")
def home():
    return {
        "message": "🏦 Banking Chatbot API is running!",
        "docs": "/docs",
        "chat_endpoint": "/chat"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "groq-llama-3.1"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main endpoint. Receives a question, returns an AI answer.
    
    Example request body:
    {
        "session_id": "user-abc-123",
        "question": "How do I reset my ATM PIN?"
    }
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Create new chain for this session (first message)
    if request.session_id not in session_chains:
        session_chains[request.session_id] = create_qa_chain()
    
    qa_chain = session_chains[request.session_id]
    result = get_answer(qa_chain, request.question)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=request.session_id
    )

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear a user's chat history"""
    if session_id in session_chains:
        del session_chains[session_id]
    return {"message": f"Session {session_id} cleared"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)