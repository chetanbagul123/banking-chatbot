"""
RAG = Retrieval Augmented Generation

Flow:
1. User asks: "How do I reset my ATM PIN?"
2. RETRIEVE: Search ChromaDB for the most relevant text chunks
3. AUGMENT: Add those chunks to the prompt as context
4. GENERATE: Groq/Llama reads the context and writes a good answer
"""

from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from backend.document_loader import get_embedding_model, load_vector_store
from backend.config import GROQ_API_KEY, GROQ_MODEL

# This prompt tells the AI exactly how to behave
SYSTEM_PROMPT_TEMPLATE = """You are a helpful and professional customer service assistant for a digital bank.

IMPORTANT RULES:
- Use ONLY the provided context to answer questions
- If the answer is not in the context, say: "I don't have that specific information. Please contact our support team or visit the nearest branch."
- Never make up information, interest rates, or policies
- Be polite, clear, and concise
- Keep answers under 150 words unless the question needs more detail

Context from our banking documents:
{context}

Previous conversation:
{chat_history}

Customer's question: {question}

Your helpful answer:"""

def create_qa_chain():
    """
    Creates the full RAG pipeline.
    Called once when the server starts.
    Returns a chain that can answer banking questions.
    """
    print("🔄 Setting up RAG pipeline...")
    
    # Step 1: Load the document database
    embeddings = get_embedding_model()
    vector_store = load_vector_store(embeddings)
    
    # Step 2: Set up the retriever
    # k=4 means: find the 4 most relevant document chunks for each question
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Step 3: Set up Groq LLM (FREE)
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.1,      # Low = more factual, less creative (good for banking)
        max_tokens=512
    )
    
    # Step 4: Memory — remembers last 5 exchanges
    # So if user asks "What about its interest rate?" after asking about home loans,
    # the bot understands "its" refers to home loans
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5
    )
    
    # Step 5: Build the custom prompt
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_PROMPT_TEMPLATE
    )
    
    # Step 6: Combine everything into one chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False
    )
    
    print("✅ RAG pipeline ready!")
    return qa_chain

def get_answer(qa_chain, question: str) -> dict:
    """
    Takes a user question → runs through RAG pipeline → returns answer.
    """
    try:
        result = qa_chain.invoke({"question": question})
        
        answer = result.get("answer", "I couldn't generate a response.")
        
        # Collect source file names (which documents were used)
        sources = []
        for doc in result.get("source_documents", []):
            source = doc.metadata.get("source", "")
            # Clean up the path to show just filename
            filename = source.split("/")[-1].split("\\")[-1]
            if filename and filename not in sources:
                sources.append(filename)
        
        return {
            "answer": answer,
            "sources": sources,
            "success": True
        }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "answer": "I'm experiencing a technical issue. Please try again in a moment.",
            "sources": [],
            "success": False,
            "error": str(e)
        }