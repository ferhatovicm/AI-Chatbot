#from langchain_openai import OpenAI  # Updated import
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from config.config import Config

class SimpleRAG:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=Config.GOOGLE_API_KEY, temperature=0.7)
        self.qa_chain = None
        
    def setup_rag_system(self, documents_path: str, vector_store_path: str):
        """Initialize the complete RAG system"""
        # Try to load existing vector store
        if not self.vector_store.load_vector_store(vector_store_path):
            print("Creating new vector store...")
            # Process documents and create vector store
            chunks = self.doc_processor.process_documents(documents_path)
            self.vector_store.create_vector_store(chunks)
            self.vector_store.save_vector_store(vector_store_path)
        
        # Create retriever
        retriever = self.vector_store.get_retriever()
        
        # Create custom prompt template
        prompt_template = """Use the following context to answer the question. 
        If you cannot answer based on the context, say so clearly.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def ask_question(self, question: str):
        """Ask a question and get answer with sources"""
        if not self.qa_chain:
            return "RAG system not initialized"
        
        result = self.qa_chain({"query": question})
        
        response = {
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "Unknown") 
                       for doc in result["source_documents"]]
        }
        
        return response
