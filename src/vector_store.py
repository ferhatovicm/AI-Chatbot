import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config.config import Config

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        self.vector_store = None
    
    def create_vector_store(self, documents):
        """Create FAISS vector store from documents"""
        self.vector_store = FAISS.from_documents(
            documents, 
            self.embeddings
        )
        return self.vector_store
    
    def save_vector_store(self, path: str):
        """Save vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"Vector store saved to {path}")
    
    def load_vector_store(self, path: str):
        """Load vector store from disk"""
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Vector store loaded from {path}")
            return True
        return False
    
    def get_retriever(self, k: int = Config.TOP_K_RETRIEVAL):
        """Get retriever for similarity search"""
        if self.vector_store:
            return self.vector_store.as_retriever(search_kwargs={"k": k})
        return None
