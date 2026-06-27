import os
from dotenv import load_dotenv

# Loads your .env file so API key is available
load_dotenv()

# Groq API key (your only key needed)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq model to use — Llama 3.1 is free and very capable
GROQ_MODEL = "llama-3.1-8b-instant"  # Fast and free
# Other options:
# "llama-3.1-70b-versatile"  — smarter but slightly slower
# "mixtral-8x7b-32768"       — also good

# HuggingFace embedding model (downloads once, runs locally — FREE)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# This is a small, fast model. Good for our use case.

# Where to store the document database
CHROMA_DB_PATH = "./chroma_db"

# Documents folder
DOCUMENTS_PATH = "./data/documents"