import os
import sys
import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# รองรับ Thai ใน Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DB_DIR = "./chroma_db"
DOCS_DIR = "./Docs"

def build_db():
    print("Finding PDF files...")
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    if not pdf_files:
        print("No PDF files found.")
        return

    documents = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        loader = PyMuPDFLoader(pdf_file)
        documents.extend(loader.load())

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    print(f"Total chunks to embed: {len(splits)}")

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Creating Chroma vectorstore...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    batch_size = 20
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        print(f"Adding batch {i//batch_size + 1}/{(len(splits)+batch_size-1)//batch_size}...")
        vectorstore.add_documents(batch)

    print("Successfully built vector DB!")

if __name__ == "__main__":
    build_db()
