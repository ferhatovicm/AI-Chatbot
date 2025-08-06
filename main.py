from src.rag_system import SimpleRAG
from config.config import Config

def main():
    # Initialize RAG system
    rag = SimpleRAG()
    
    # Setup the system
    rag.setup_rag_system(
        documents_path=Config.DOCUMENTS_PATH,
        vector_store_path=Config.VECTOR_STORE_PATH
    )
    
    # Interactive Q&A loop
    print("RAG System ready! Ask questions (type 'quit' to exit)")
    
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'quit':
            break
            
        response = rag.ask_question(question)
        print(f"\nAnswer: {response['answer']}")
        print(f"Sources: {', '.join(response['sources'])}")

if __name__ == "__main__":
    main()
