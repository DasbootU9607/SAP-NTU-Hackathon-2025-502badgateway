from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load Vector DB
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# Setup RAG
def get_rag_response(query):
    docs = db.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Answer based on the following context:\n{context}\n\nQuestion: {query}"
    
    # Use OpenAI / Ollama here
    from langchain.llms import OpenAI
    llm = OpenAI(api_key="YOUR_KEY")  # or use Ollama
    response = llm(prompt)
    return response