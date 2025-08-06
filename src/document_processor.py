from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.config import Config

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self, path: str):
        """Load all text documents from directory"""
        loader = DirectoryLoader(path, glob="*.txt", loader_cls=lambda p: TextLoader(p, encoding="utf-8"))
        documents = loader.load()
        return documents
    
    def split_documents(self, documents):
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def process_documents(self, path: str):
        """Complete document processing pipeline"""
        documents = self.load_documents(path)
        chunks = self.split_documents(documents)
        print(f"Processed {len(documents)} documents into {len(chunks)} chunks")
        return chunks
