# rag_setup.py
from document_processor import document_processor
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os

def setup_rag_system():
    # 1. Process all documents
    print("=" * 50)
    print("Starting company document processing...")
    print("=" * 50)
    
    data_directory = "./company_data"  # Company data directory
    documents = document_processor.process_directory(data_directory)
    
    if not documents:
        print("No processable documents found, please check the company_data directory")
        return
    
    print(f"\nTotal generated {len(documents)} document chunks")
    
    # 2. Create vector database
    print("\nCreating vector database...")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=OllamaEmbeddings(model="llama3", base_url="http://localhost:11434"),
        persist_directory="./chroma_db_company"
    )
    vector_db.persist()
    
    print("✅ RAG system setup completed!")
    print(f"📊 Knowledge base contains {len(documents)} document chunks")
    print(f"💾 Vector database saved to: ./chroma_db_company")

if __name__ == "__main__":
    setup_rag_system()