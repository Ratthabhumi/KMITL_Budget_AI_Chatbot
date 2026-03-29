import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
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
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"ไม่สามารถโหลดไฟล์ {pdf_file} ได้: {e}")
            
        progress_bar.progress((i + 1) / len(pdf_files), text=f"กำลังโหลด... {os.path.basename(pdf_file)}")
            
    # 2. ตัดแบ่งข้อความ (Chunking) เพื่อให้โมเดลประมวลผลได้ดีขึ้น
    progress_bar.progress(0.9, text="กำลังตัดแบ่งข้อความ (Splitting)... อาจใช้เวลาสักครู่")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # 3. สร้าง Embeddings และจัดเก็บลงฐานข้อมูลเรื่อยๆ
    progress_bar.progress(0.95, text="กำลังฝังเวกเตอร์ (Embedding) ไปยัง ChromaDB... (กระบวนการนี้จะคุยกับ Gemini API และอาจใช้เวลาหลายนาที)")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # วิธีนี้จะสร้าง folder ชื่อ chroma_db เพื่อเก็บข้อมูลไว้เปิดครั้งต่อไปได้เลย
    import time
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    batch_size = 20
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        vectorstore.add_documents(documents=batch)
        if i + batch_size < len(splits):
            time.sleep(15) # หน่วงเวลาเล็กน้อยเพื่อป้องกัน API ดักเตะ (Rate Limit / 429 Error)
    
    progress_bar.empty()
    st.success(f"ดำเนินการเสร็จสิ้น! โหลดเอกสารทั้งหมด {len(pdf_files)} ไฟล์ เรียบร้อยแล้ว")
    return True

def get_qa_chain(api_key):
    """สร้าง Chain สำหรับตอบคำถามอ้างอิงจาก Vector DB ที่มีอยู่"""
    
    if not os.path.exists(DB_DIR):
        return None
        
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # โหลดจากฐานข้อมูลเดิมที่เคยบันทึกไว้
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # ให้ค้นหา 5 เอกสารที่เกี่ยวข้องที่สุด (ปรับแต่งได้ตามความเหมาะสม)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # ตั้งค่าโมเดล Gemini สำหรับใช้อ่านบริบทและตอบคำถาม
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0) 
    
    # กำหนด Prompt (หน้าที่ของ AI) ให้ชัดเจน
    system_prompt = (
        "คุณคือ AI ผู้ช่วยอัจฉริยะของสถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง (สจล.) "
        "หน้าที่หลักของคุณคือการตอบคำถามเกี่ยวกับ 'ระเบียบการเบิกจ่ายงบประมาณและการจัดซื้อจัดจ้าง' ให้กับบุคลากรและนักศึกษาอย่างถูกต้องและแม่นยำที่สุด\n\n"
        "คำสั่งสำคัญ:\n"
        "1. ให้ใช้ข้อมูลอ้างอิงที่ได้รับมาด้านล่างนี้ (Context) เพื่อตอบคำถามผู้ใช้เท่านั้น\n"
        "2. หากไม่มีข้อมูลใน Context ที่เกี่ยวข้อง ให้ตอบตามตรงอย่างสุภาพว่า 'ขออภัยครับ/ค่ะ ไม่พบข้อมูลในระเบียบการที่ให้มา' ห้ามมั่วข้อมูล (Hallucination) หรือคิดเอาเองเด็ดขาด\n"
        "3. หากอ้างอิงข้อมูลได้ ให้อธิบายเป็นข้อๆ ให้ผู้ใช้อ่านเข้าใจง่าย เป็นภาษาไทยที่เป็นทางการแต่เป็นมิตร\n"
        "4. หากสามารถระบุชื่อเอกสารหรือคู่มือที่อ้างอิงตาม Context ได้ให้ระบุไว้ด้วยก็จะดีมาก\n\n"
        "=== ข้อมูลอ้างอิงจากระบบ (Context) ===\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # สร้าง Chain (ประกอบร่างโมเดลกับ Prompt เข้าด้วยกัน)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain
