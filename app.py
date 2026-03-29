import streamlit as st
import os
import json
from datetime import datetime
import google.generativeai as genai
from rag_pipeline import initialize_vector_db, get_qa_chain

# การตั้งค่าหน้ากระดาษ (Page Configuration)
st.set_page_config(
    page_title="KMITL Budget AI",
    page_icon="🎓",
    layout="wide"
)

# === ฟังก์ชันบันทึก Feedback ===
def save_feedback(question, answer, is_upvote):
    feedback_file = "feedback_log.json"
    data = []
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
            
    data.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "feedback": "👍 ดี/ถูกต้อง" if is_upvote else "👎 ไม่ถูกต้อง/มั่ว"
    })
    
    with open(feedback_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ส่วนของ Sidebar สำหรับการตั้งค่า
with st.sidebar:
    st.title("⚙️ การตั้งค่า")
    api_key_input = st.text_input("ระบุ Gemini API Key (หากไม่มีใน secrets.toml):", type="password")
    
    try:
        # พยายามดึง Key จาก secrets.toml ก่อน
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"].strip() # .strip() ช่วยตัด \n ออกให้ด้วยครับ
        else:
            # ถ้าไม่มีในไฟล์ ค่อยไปดูที่ Sidebar
            api_key = api_key_input 

        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemma-3-12b-it')
        else:
            st.warning("⚠️ ไม่พบ API Key กรุณาตรวจสอบไฟล์ secrets.toml หรือระบุที่ Sidebar")
            st.stop()
            
    except Exception as e:
        st.error(f"Error connecting to Gemini: {e}")
        st.stop()
        
    gemini_api_key = api_key # ให้ตัวแปรเดิมยังทำงานได้ต่อเนื่อง
    
    st.divider()
    st.markdown("""
    ### เกี่ยวกับโปรเจกต์
    ระบบนี้ใช้ AI ช่วยตอบคำถามเกี่ยวกับ
    **ระเบียบการเบิกจ่ายงบประมาณ KMITL**
    พัฒนาโดยนักศึกษาคณะวิศวกรรมศาสตร์
    """)
    if gemini_api_key:
        st.divider()
        st.write("⚙️ โหมดผู้ดูแลระบบ")
        if st.button("🔄 โหลดเอกสารจากโฟลเดอร์ Docs เป็นฐานข้อมูล", help="กดปุ่มนี้เพื่ออ่านไฟล์ PDF ทั้งหมดไปทำ RAG (ใช้เวลาสักพัก)"):
            initialize_vector_db(gemini_api_key)

# ส่วนหัวของหน้าเว็บหลัก
st.title("🎓 KMITL Budget AI Chatbot")
st.subheader("ระบบสนับสนุนการตอบคำถามด้านการเบิกจ่ายงบประมาณ")

# ตัวอย่างสถานะการเชื่อมต่อ (Mockup)
if not gemini_api_key:
    st.warning("กรุณาใส่ API Key ที่ Sidebar ด้านซ้ายเพื่อเริ่มต้นใช้งาน")
else:
    st.success("API Key พร้อมใช้งานแล้ว")

# ส่วนแสดงประวัติการแชท (Chat History)
if "messages" not in st.session_state:
    st.session_state.messages = []

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # ตรวจสอบว่าในประวัติมีเก็บ source_documents ไว้หรือไม่ (สำหรับ AI เท่านั้น)
        if message["role"] == "assistant":
            if "source_documents" in message and message["source_documents"]:
                with st.expander("📄 ดูเอกสารอ้างอิงต้นฉบับ"):
                    for idx, doc in enumerate(message["source_documents"]):
                        source_name = os.path.basename(doc.metadata.get('source', 'ไม่ระบุชื่อไฟล์'))
                        page = doc.metadata.get('page', 0)
                        if isinstance(page, int):
                            page += 1
                        st.markdown(f"**อ้างอิงที่ {idx+1}:** ไฟล์ `{source_name}` (หน้าที่ {page})")
                        st.info(doc.page_content)
                        
            # ปุ่ม Feedback (ขึ้นบรรทัดใหม่หลังคำตอบและเอกสาร)
            # ดึงคำถามที่คู่กับคำตอบนี้ (มักจะเป็น message ก่อนหน้า)
            prev_question = st.session_state.messages[i-1]["content"] if i > 0 else "ไม่ทราบคำถาม"
            
            c1, c2, c3 = st.columns([1, 1, 8])
            with c1:
                if st.button("👍 ดี", key=f"upvote_{i}"):
                    save_feedback(prev_question, message["content"], True)
                    st.toast("ขอบคุณสำหรับข้อเสนอแนะ! บันทึกแล้วค่ะ 👍")
            with c2:
                if st.button("👎 แย่", key=f"downvote_{i}"):
                    save_feedback(prev_question, message["content"], False)
                    st.toast("ขอบคุณสำหรับข้อเสนอแนะ! เราจะนำไปปรับปรุงค่ะ 👎")

# ส่วนรับข้อความจากผู้ใช้ (Chat Input)
if prompt := st.chat_input("สอบถามเรื่องการเบิกจ่ายได้ที่นี่..."):
    # แสดงข้อความของผู้ใช้
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ส่วนประมวลผลของ AI ด้วย RAG (เชื่อมต่อ Gemini)
    with st.chat_message("assistant"):
        if not gemini_api_key:
            response = "⚠️ กรุณากรอก API Key ที่ Sidebar ด้านซ้ายก่อนถามคำถามค่ะ"
            st.warning(response)
        else:
            with st.spinner("AI กำลังค้นคว้าระเบียบการและค้นหาคำตอบ..."):
                qa_chain = get_qa_chain(gemini_api_key)
                if qa_chain is None:
                    response = "⚠️ ยังขาดฐานข้อมูลระเบียบการ! รบกวนกดปุ่ม 'โหลดเอกสาร' ด้านซ้ายมือก่อนใช้งานครั้งแรกค่ะ"
                    st.error(response)
                else:
                    try:
                        result = qa_chain.invoke({"input": prompt})
                        response = result["answer"]
                        st.markdown(response)
                        
                        source_docs = result.get("context", [])
                        if source_docs:
                            with st.expander("📄 ดูเอกสารอ้างอิงต้นฉบับ (Source Documents)"):
                                for i, doc in enumerate(source_docs):
                                    source_name = os.path.basename(doc.metadata.get('source', 'ไม่ระบุชื่อไฟล์'))
                                    page = doc.metadata.get('page', 0)
                                    if isinstance(page, int):
                                        page += 1
                                    st.markdown(f"**อ้างอิงที่ {i+1}:** ไฟล์ `{source_name}` (หน้าที่ {page})")
                                    st.info(doc.page_content)
                                    
                    except Exception as e:
                        response = f"**เกิดข้อผิดพลาดระหว่างใช้ AI:** {e}"
                        st.error(response)
                        source_docs = []

    st.session_state.messages.append({"role": "assistant", "content": response, "source_documents": source_docs if 'source_docs' in locals() else []})

# ฟังก์ชันสำหรับอัปโหลดไฟล์ระเบียบการ (Future Work)
with st.expander("📁 อัปโหลดระเบียบการเพิ่มเติม (สำหรับ Admin)"):
    uploaded_file = st.file_type_uploader = st.file_uploader("เลือกไฟล์ PDF ระเบียบการ", type="pdf")
    if uploaded_file:
        st.write("ไฟล์ถูกอัปโหลดแล้ว: ", uploaded_file.name)