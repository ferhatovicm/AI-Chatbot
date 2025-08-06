import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

class Config:
    # Vector store settings
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50
    TOP_K_RETRIEVAL = 3
    
    # Paths
    DOCUMENTS_PATH = "data/documents"
    VECTOR_STORE_PATH = "data/vector_store"
    
    # LLM settings (choose your provider)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    MODEL_NAME = "models/gemini-2.0-flash"  # or your chosen model
    
    # Embedding model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
