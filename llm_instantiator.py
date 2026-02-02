import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

def get_llm(model_type: str):
    """Initialize the LLM based on user selection."""
    if model_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        return ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)
    
    elif model_type == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")