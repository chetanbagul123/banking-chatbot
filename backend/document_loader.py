"""
This file handles:
1. Reading all documents from the data/documents folder
2. Splitting them into small chunks
3. Converting text to numbers (embeddings) using FREE HuggingFace model
4. Saving everything into ChromaDB
"""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from backend.config import CHROMA_DB_PATH, DOCUMENTS_PATH, EMBEDDING_MODEL

def get_embedding_model():
    """
    Load the FREE HuggingFace embedding model.
    
    First time: downloads ~90MB model to your computer.
    After that: loads from your computer instantly. No internet needed.
    No API calls, no cost ever.
    """
    print("🔄 Loading embedding model (first time takes 1-2 minutes to download)...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # Use CPU (works on any computer)
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print("✅ Embedding model ready!")
    return embeddings

def load_documents():
    """Load all text and PDF files from the documents folder"""
    documents = []
    
    for filename in os.listdir(DOCUMENTS_PATH):
        filepath = os.path.join(DOCUMENTS_PATH, filename)
        
        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())
            print(f"  📄 Loaded: {filename}")
            
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            print(f"  📄 Loaded: {filename}")
    
    print(f"✅ Total documents loaded: {len(documents)}")
    return documents

def split_documents(documents):
    """
    Split large documents into smaller chunks.
    
    Why? The AI can't read 100 pages at once. We find the
    most relevant 3-4 paragraphs and send just those.
    
    chunk_size=500: each piece is ~500 characters
    chunk_overlap=50: pieces overlap so no info is cut at edges
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks, embeddings):
    """
    Convert text chunks into numbers and store in ChromaDB.
    ChromaDB saves to your disk so you don't redo this every time.
    """
    print("🔄 Creating vector store (may take 1-2 minutes)...")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    
    print("✅ Vector store created and saved to disk!")
    return vector_store

def load_vector_store(embeddings):
    """Load existing ChromaDB from disk (fast, used after first-time setup)"""
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )

def initialize_knowledge_base():
    """
    Run this ONCE to process all documents.
    After this, your chroma_db/ folder has everything saved.
    """
    print("\n🚀 Initializing Banking Knowledge Base...\n")
    
    embeddings = get_embedding_model()
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks, embeddings)
    
    print("\n🎉 Knowledge base is ready! You can now start the server.\n")

if __name__ == "__main__":
    initialize_knowledge_base()