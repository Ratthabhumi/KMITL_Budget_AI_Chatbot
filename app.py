import streamlit as st
import os
from rag_pipeline import initialize_vector_db, get_qa_chain

# การตั้งค่าหน้ากระดาษ (Page Configuration)
st.set_page_config(
    page_title="KMITL Budget AI",
    page_icon="🎓",
    layout="wide"
)

# ส่วนของ Sidebar สำหรับการตั้งค่า
with st.sidebar:
    st.title("⚙️ การตั้งค่า")
    # ดึง API Key จากไฟล์ .streamlit/secrets.toml โดยอัตโนมัติ (ถ้ามี)
    # ถ้าไม่มีไฟล์ secrets ก็ให้ผู้ใช้กรอกเองแทน
    if "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        st.success("🔑 API Key โหลดจาก secrets อัตโนมัติแล้ว")
    else:
        gemini_api_key = st.text_input("ระบุ Gemini API Key:", type="password")
        st.info("คุณสามารถขอ API Key ได้ที่ [Google AI Studio](https://aistudio.google.com/)")
    
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
                    except Exception as e:
                        response = f"**เกิดข้อผิดพลาดระหว่างใช้ AI:** {e}"
                        st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ฟังก์ชันสำหรับอัปโหลดไฟล์ระเบียบการ (Future Work)
with st.expander("📁 อัปโหลดระเบียบการเพิ่มเติม (สำหรับ Admin)"):
    uploaded_file = st.file_type_uploader = st.file_uploader("เลือกไฟล์ PDF ระเบียบการ", type="pdf")
    if uploaded_file:
        st.write("ไฟล์ถูกอัปโหลดแล้ว: ", uploaded_file.name)