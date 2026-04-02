import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

DB_DIR = "./chroma_db"
DOCS_DIR = "./Docs"

def initialize_vector_db(api_key):
    """อ่านไฟล์ PDF ทั้งหมดในโฟลเดอร์ Docs และสร้างฐานข้อมูลเวกเตอร์ ChromaDB"""
    
    # ตั้งค่า API Key ให้กับ environment (Langchain จะนำไปใช้โดยอัตโนมัติ)
    os.environ["GOOGLE_API_KEY"] = api_key
    
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    if not pdf_files:
        st.error(f"ไม่พบไฟล์ PDF ในโฟลเดอร์ {DOCS_DIR}")
        return False
        
    documents = []
    
    # สร้างตัวแสดงสถานะ
    progress_bar = st.progress(0, text="กำลังเตรียมไฟล์...")
    
    # 1. โหลดเอกสาร PDF
    for i, pdf_file in enumerate(pdf_files):
        try:
            loader = PyMuPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"ไม่สามารถโหลดไฟล์ {pdf_file} ได้: {e}")
            
        progress_bar.progress((i + 1) / len(pdf_files), text=f"กำลังโหลด... {os.path.basename(pdf_file)}")
            
    # 2. ตัดแบ่งข้อความ (Chunking) เพื่อให้โมเดลประมวลผลได้ดีขึ้น
    progress_bar.progress(0.9, text="กำลังตัดแบ่งข้อความ (Splitting)... อาจใช้เวลาสักครู่")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    # 3. สร้าง Embeddings ด้วย Local Model (ไม่ต้องใช้ API ของ Google อีกต่อไป)
    progress_bar.progress(0.95, text="กำลังโหลดโมเดลภาษาไทย (Local Embedding) และบันทึกลง ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # วิธีนี้จะสร้าง folder ชื่อ chroma_db เพื่อเก็บข้อมูลไว้เปิดครั้งต่อไปได้เลย
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    batch_size = 20
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        vectorstore.add_documents(documents=batch)
    
    progress_bar.empty()
    st.success(f"ดำเนินการเสร็จสิ้น! โหลดเอกสารทั้งหมด {len(pdf_files)} ไฟล์ เรียบร้อยแล้ว")
    return True

def get_qa_chain(api_key):
    """สร้าง Chain สำหรับตอบคำถามอ้างอิงจาก Vector DB ที่มีอยู่"""
    
    if not os.path.exists(DB_DIR):
        return None
        
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # โหลดจากฐานข้อมูลเดิมที่เคยบันทึกไว้
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    # สามารถใช้ gemma-3-4b-it ได้ หรือถ้าอยากให้ฉลาดขึ้นมากแบบฟรี 15 ครั้ง/นาที 
    # แนะนำให้เปลี่ยนตัวเลขเป็น "gemini-1.5-flash"
    llm = ChatGoogleGenerativeAI(model="gemma-3-4b-it", temperature=0)
    
    # จัดหน้ากระดาษ Prompt ใหม่ให้โมเดลตัวเล็กๆ เข้าใจง่ายขึ้นโดยใช้ XML tags
    combined_prompt = (
        "คุณคือ AI ผู้ช่วยอัจฉริยะของสถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง (สจล.)\n"
        "หน้าที่ของคุณคือตอบคำถามเกี่ยวกับ 'ระเบียบงบประมาณและการจัดซื้อจัดจ้าง' ด้วยความถูกต้องและแม่นยำ\n\n"
        "คำสั่ง:\n"
        "1. ตอบคำถามโดยอ้างอิงจากข้อมูลใน <context> เท่านั้น ห้ามคิดเอง\n"
        "2. หากข้อมูลใน <context> ไม่เกี่ยวหรือไม่มีคำตอบ ให้ตอบว่า 'ขออภัย ไม่พบข้อมูลในระเบียบการครับ/ค่ะ'\n"
        "3. อธิบายให้เข้าใจง่าย เป็นข้อๆ และเป็นทางการ\n\n"
        "<context>\n"
        "{context}\n"
        "</context>\n\n"
        "คำถาม: {input}\n"
        "คำตอบ:"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", combined_prompt),
    ])
    
    # สร้าง Chain (ประกอบร่างโมเดลกับ Prompt เข้าด้วยกัน)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain
