import streamlit as st
import os
import json
from datetime import datetime
import pandas as pd
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
    
    # 📌 ส่วนเลือกหน้าจอการทำงาน (Navigation)
    st.markdown("### 📌 เมนูระบบ")
    page = st.radio("เลือกหน้าต่างการทำงาน:", ["💬 แชทบอท (Chatbot)", "📸 ตรวจสอบใบเสร็จ (Receipt Verification)", "📊 แดชบอร์ดผู้ดูแล (Admin)"])
    st.divider()
    
    api_key_input = st.text_input("ระบุ Gemini API Key (หากไม่มีใน secrets.toml):", type="password")
    
    try:
        # พยายามดึง Key จาก secrets.toml ก่อน
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"].strip()
        else:
            api_key = api_key_input 

        if not api_key:
            st.warning("⚠️ ไม่พบ API Key กรุณาตรวจสอบไฟล์ secrets.toml หรือระบุที่ Sidebar")
            st.stop()
            
    except Exception as e:
        st.error(f"Error connecting to Gemini: {e}")
        st.stop()
        
    gemini_api_key = api_key
    
    st.divider()
    st.markdown("""
    ### เกี่ยวกับโปรเจกต์
    ระบบนี้ใช้ AI ช่วยตอบคำถามเกี่ยวกับ
    **ระเบียบการเบิกจ่ายงบประมาณ KMITL**
    """)
    if gemini_api_key:
        st.divider()
        st.write("⚙️ จัดการฐานข้อมูล (DB)")
        if st.button("🔄 โหลดเอกสารจากโฟลเดอร์ Docs เป็นฐานข้อมูล", help="สกัดและทำ Vector DB"):
            initialize_vector_db(gemini_api_key)

# ==========================================
# 💬 หน้า 1: Chatbot View
# ==========================================
if page == "💬 แชทบอท (Chatbot)":
    st.title("🎓 KMITL Budget AI Chatbot")
    st.subheader("ระบบสนับสนุนการตอบคำถามด้านการเบิกจ่ายงบประมาณ")

    if not gemini_api_key:
        st.warning("กรุณาใส่ API Key ที่ Sidebar ด้านซ้ายเพื่อเริ่มต้นใช้งาน")
    else:
        st.success("API Key พร้อมใช้งานแล้ว")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant":
                if "source_documents" in message and message["source_documents"]:
                    with st.expander("📄 ดูเอกสารอ้างอิงต้นฉบับ"):
                        for idx, doc in enumerate(message["source_documents"]):
                            source_name = os.path.basename(doc.metadata.get('source', 'ไม่ระบุชื่อไฟล์'))
                            page_num = doc.metadata.get('page', 0)
                            if isinstance(page_num, int):
                                page_num += 1
                            st.markdown(f"**อ้างอิงที่ {idx+1}:** ไฟล์ `{source_name}` (หน้าที่ {page_num})")
                            st.info(doc.page_content)
                            
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

    if prompt := st.chat_input("สอบถามเรื่องการเบิกจ่ายได้ที่นี่..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not gemini_api_key:
                response = "⚠️ กรุณากรอก API Key ที่ Sidebar ด้านซ้ายก่อนถามคำถามค่ะ"
                st.warning(response)
            else:
                with st.spinner("AI กำลังค้นคว้าระเบียบการและค้นหาคำตอบ..."):
                    qa_chain = get_qa_chain(gemini_api_key)
                    if qa_chain is None:
                        response = "⚠️ ยังขาดฐานข้อมูลระเบียบการ! รบกวนกดปุ่ม 'โหลดเอกสาร' ด้านซ้ายมือก่อน"
                        st.error(response)
                    else:
                        try:
                            result = qa_chain.invoke({"input": prompt})
                            response = result["answer"]
                            st.markdown(response)
                            source_docs = result.get("context", [])
                            if source_docs:
                                with st.expander("📄 ดูเอกสารอ้างอิงต้นฉบับ (Source Documents)"):
                                    for idx, doc in enumerate(source_docs):
                                        source_name = os.path.basename(doc.metadata.get('source', 'ไม่ระบุชื่อไฟล์'))
                                        page_num = doc.metadata.get('page', 0)
                                        if isinstance(page_num, int):
                                            page_num += 1
                                        st.markdown(f"**อ้างอิงที่ {idx+1}:** ไฟล์ `{source_name}` (หน้าที่ {page_num})")
                                        st.info(doc.page_content)
                        except Exception as e:
                            response = f"**เกิดข้อผิดพลาดระหว่างใช้ AI:** {e}"
                            st.error(response)
                            source_docs = []

        st.session_state.messages.append({"role": "assistant", "content": response, "source_documents": source_docs if 'source_docs' in locals() else []})

# ==========================================
# 📸 หน้า 1.5: Receipt Verification
# ==========================================
elif page == "📸 ตรวจสอบใบเสร็จ (Receipt Verification)":
    st.title("📸 ตรวจสอบใบเสร็จด้วย AI (Receipt Verification)")
    st.markdown("ระบบจะใช้ AI (OCR) อ่านข้อมูลจากรูปภาพใบเสร็จเพื่อตรวจสอบความถูกต้องและเช็กเงื่อนไขกับระเบียบเบิกจ่ายของ KMITL แบบอัตโนมัติ")
    
    if not gemini_api_key:
        st.warning("⚠️ กรุณากรอก API Key ที่ Sidebar ด้านซ้ายก่อนค่ะ")
    else:
        c1, c2 = st.columns(2)
        with c1:
            uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์รูปภาพใบเสร็จ", type=["png", "jpg", "jpeg"])
        with c2:
            camera_file = st.camera_input("📷 หรือเปิดกล้องถ่ายใบเสร็จ")
            
        target_image = camera_file if camera_file else uploaded_file
        
        if target_image:
            st.image(target_image, caption="รายการเอกสารที่แนบ", width=400)
            
            if st.button("🔍 ตรวจสอบและประมวลผล", type="primary"):
                from ocr_pipeline import extract_receipt_data, verify_receipt_rules
                
                with st.spinner("⏳ กำลังใช้ Gemini 1.5 Flash (OCR Engine) สกัดตัวอักษรและโครงสร้างใบเสร็จ..."):
                    img_bytes = target_image.getvalue()
                    ocr_data = extract_receipt_data(img_bytes, gemini_api_key)
                    
                if "error" in ocr_data and "raw_content" not in ocr_data:
                    st.error(f"❌ ระบบประมวลผลภาพผิดพลาด: {ocr_data['error']}")
                else:
                    st.success("✅ สกัดแยกหมวดหมู่ข้อมูลเอกสารสำเร็จ")
                    st.markdown("### 📝 ข้อมูลที่สกัดได้จากใบเสร็จ (Extracted Data)")
                    st.json(ocr_data)
                    
                    st.markdown("### ⚖️ ผลการวิเคราะห์และตรวจสอบความสมบูรณ์")
                    with st.spinner("⏳ กำลังเปรียบเทียบข้อมูลกับฐานข้อมูลระเบียบการเบิกจ่าย (RAG Verification)..."):
                        qa_chain = get_qa_chain(gemini_api_key)
                        if qa_chain is None:
                            st.error("⚠️ ยังขาดฐานข้อมูลระเบียบการ! รบกวนกดปุ่ม 'โหลดเอกสาร' ด้านซ้ายมือก่อน")
                        else:
                            ai_analysis = verify_receipt_rules(qa_chain, ocr_data)
                            st.info(ai_analysis)
                            
                            st.markdown("---")
                            st.markdown("### 📬 จำลองระบบอนุมัติอัตโนมัติ (Automated Workflow)")
                            st.button("✅ ยืนยันความสมบูรณ์ และจัดส่งเข้าสู่ระบบ DMS ของสถาบัน", type="primary")

# ==========================================
# 📊 หน้า 2: Admin Dashboard
# ==========================================
elif page == "📊 แดชบอร์ดผู้ดูแล (Admin)":
    st.title("📊 แดชบอร์ดประเมินการทำงาน (Admin Dashboard)")
    st.markdown("ตรวจสอบสถิติความแม่นยำของโมเดล (Evaluation) และความคิดเห็นของผู้ใช้งานจริง (User Feedback)")

    # โหลดข้อมูลต่างๆ
    retrieval_data, eval_data, feedback_data = {}, [], []
    retrieval_file = "retrieval_eval_report.json"
    eval_file = "evaluation_report.json"
    feedback_file = "feedback_log.json"

    if os.path.exists(retrieval_file):
        with open(retrieval_file, "r", encoding="utf-8") as f:
            retrieval_data = json.load(f)
    if os.path.exists(eval_file):
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    if os.path.exists(feedback_file):
        with open(feedback_file, "r", encoding="utf-8") as f:
            feedback_data = json.load(f)

    st.markdown("### 🏆 1. ภาพรวมความแม่นยำของระบบค้นหา (Retrieval Metrics)")
    if retrieval_data and "overall_metrics" in retrieval_data:
        metrics = retrieval_data["overall_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📌 ข้อสอบทั้งหมด", f"{retrieval_data.get('total_questions', 0)} ข้อ")
        c2.metric("🎯 Avg Recall@K", f"{metrics.get('Avg_Recall@6', 0):.4f}")
        c3.metric("📏 Avg Precision@K", f"{metrics.get('Avg_Precision@6', 0):.4f}")
        c4.metric("🥇 Avg MRR", f"{metrics.get('Avg_MRR', 0):.4f}")
    else:
        st.info("ยังไม่มีข้อมูล Retrieval Evaluation กรุณารัน `retrieval_eval.py`")

    st.divider()

    st.markdown("### ⚖️ 2. ผลการประเมินกรรมการ AI (LLM-as-a-Judge Scores)")
    has_scores = False
    f_scores, r_scores, c_scores = [], [], []

    for item in eval_data:
        scores = item.get("evaluation_scores")
        if scores and not isinstance(scores, str) and "error" not in scores: # ตรวจสอบเผื่อมี Error
            f_scores.append(scores.get("Faithfulness", 0))
            r_scores.append(scores.get("Answer_Relevance", 0))
            c_scores.append(scores.get("Context_Precision", 0))

    if f_scores:
        has_scores = True
        avg_f = sum(f_scores) / len(f_scores)
        avg_r = sum(r_scores) / len(r_scores)
        avg_c = sum(c_scores) / len(c_scores)

        c1, c2, c3 = st.columns(3)
        c1.metric("💡 Avg Faithfulness", f"{avg_f:.2f}/5", "ไม่เกิดการมั่ว (Hallucination)")
        c2.metric("🎯 Avg Relevance", f"{avg_r:.2f}/5", "ตอบตรงประเด็น")
        c3.metric("📄 Context Precision", f"{avg_c:.2f}/5", "ข้อสอบตรงกับเอกสาร")

        # กราฟแท่งเปรียบเทียบคะแนน
        chart_data = pd.DataFrame(
            {"คะแนนเฉลี่ย": [avg_f, avg_r, avg_c]},
            index=["Faithfulness", "Answer Relevance", "Context Precision"]
        )
        st.bar_chart(chart_data)
        
        with st.expander("📝 ดูข้อสอบที่มีปัญหา (คะแนนต่ำกว่า 3)"):
            low_score_items = []
            for item in eval_data:
                sc = item.get("evaluation_scores", {})
                if sc and not isinstance(sc, str) and "error" not in sc:
                    if sc.get("Faithfulness", 5) < 3 or sc.get("Answer_Relevance", 5) < 3:
                        low_score_items.append({
                            "Question": item["question"],
                            "Faithfulness": sc.get("Faithfulness"),
                            "Relevance": sc.get("Answer_Relevance"),
                            "Reasoning": sc.get("Reasoning", "")
                        })
            if low_score_items:
                st.dataframe(pd.DataFrame(low_score_items), use_container_width=True)
            else:
                st.success("🎉 ไม่มีข้อสอบข้อใดที่ได้คะแนนความแม่นยำต่ำกว่า 3 เลย!")
    else:
        st.info("ยังไม่มีข้อมูลของ LLM-as-a-Judge ในชุดข้อมูล! (คุณสามารถรัน `python ai_evaluator.py` เพื่อให้กรรมการตรวจและให้คะแนน)")

    st.divider()

    st.markdown("### 💌 3. ความคิดเห็นของผู้ใช้งาน (User Feedback)")
    if feedback_data:
        df_feed = pd.DataFrame(feedback_data)
        feed_counts = df_feed["feedback"].value_counts()
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("ผู้ใช้ประเมินทั้งหมด", f"{len(df_feed)} ครั้ง")
            up_count = int(feed_counts.get("👍 ดี/ถูกต้อง", 0))
            down_count = int(feed_counts.get("👎 ไม่ถูกต้อง/มั่ว", 0))
            st.metric("👍 จำนวนโหวตดี", up_count)
            st.metric("👎 จำนวนโหวตแย่", down_count)
        with c2:
            st.bar_chart(feed_counts, color="#FF4B4B")
            
        with st.expander("🔍 ดูข้อความแชทที่มีคนโหวตแย่ (เพื่อนำไปปรับปรุง)"):
            bad_feedbacks = df_feed[df_feed["feedback"] == "👎 ไม่ถูกต้อง/มั่ว"]
            if not bad_feedbacks.empty:
                st.dataframe(bad_feedbacks[["timestamp", "question", "answer"]], use_container_width=True)
            else:
                st.success("🎉 ยังไม่มีการโหวตคะแนนแย่ในระบบ!")
    else:
        st.info("ยังไม่มีข้อมูล Feedback จากผู้ใช้งาน (ลองปั๊มโหวต 👍/👎 ในหน้าแชทดูสิ!)")