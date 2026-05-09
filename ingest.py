from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# 1. Setup
print("Initializing Ingestion System...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "./db"

# 2. Load ALL documents from the data folder
print("Scanning 'data' folder for documents...")
# We use DirectoryLoader to catch multiple file types
pdf_loader = DirectoryLoader('./data', glob="./*.pdf", loader_cls=PyPDFLoader)
txt_loader = DirectoryLoader('./data', glob="./*.txt", loader_cls=TextLoader)

docs = []
docs.extend(pdf_loader.load())
docs.extend(txt_loader.load())

print(f"Successfully loaded {len(docs)} documents.")

# 3. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks of knowledge.")

# 4. Save to ChromaDB
print("Saving knowledge to permanent brain (ChromaDB)...")
vectordb = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory=persist_directory
)
print("SUCCESS! Your Multi-Document Brain is ready.")
