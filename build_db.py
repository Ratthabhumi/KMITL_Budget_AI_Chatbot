import os
import time
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

DB_DIR = "./chroma_db"
DOCS_DIR = "./Docs"

# ใช้ API Key ที่แฝงมาให้
API_KEY = "AIzaSyDsXWfTx5iMZQZg2AMwonbDXL2N-4_L9z0"
os.environ["GOOGLE_API_KEY"] = API_KEY

def build_db():
    print("Finding PDF files...")
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    if not pdf_files:
        print("No PDF files found.")
        return

    documents = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Total chunks to embed: {len(splits)}")

    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    print("Creating Chroma vectorstore with rate-limit evasion...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Process in chunks to avoid 429
    batch_size = 10 
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        print(f"Adding batch {i//batch_size + 1}/{(len(splits)+batch_size-1)//batch_size}...")
        # Add to vectorstore
        vectorstore.add_documents(batch)
        # Sleep to keep requests below 100/minute. 
        # A batch of 10 might generate 10 requests. Let's wait a few seconds.
        print("Waiting 10 seconds to respect API rate limits...")
        time.sleep(10)

    print("Successfully built vector DB!")

if __name__ == "__main__":
    build_db()
