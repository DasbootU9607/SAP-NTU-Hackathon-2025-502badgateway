
from langchain.document_loaders import DirectoryLoader, TextLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os

# Load documents
loader = DirectoryLoader('./docs/', glob="**/*", loader_cls=TextLoader)  # or PDFMinerLoader
documents = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Embed & Store
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

vectorstore.persist()
print("✅ Knowledge base indexed in ChromaDB.")