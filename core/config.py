import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Streamlit secrets helper (if running inside Streamlit)
def get_secret(key: str, default=None):
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# Google API Key
GOOGLE_API_KEY = (
    get_secret("GOOGLE_GEMINI_API_KEY") 
    or get_secret("GOOGLE_API_KEY")
)

# Admin Credentials
ADMIN_USERNAME = get_secret("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD", "admin123")

# Model configurations
EMBEDDING_MODEL_NAME = "gemini-embedding-2-preview"
LLM_MODEL_NAME = "gemini-3.5-flash-lite"

def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME, 
        google_api_key=GOOGLE_API_KEY
    )

def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME, 
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
