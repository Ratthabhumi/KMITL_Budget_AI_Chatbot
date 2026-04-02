import os
import json
import time
from datetime import datetime

# พยายามโหลด st.secrets และดึง API KEY
try:
    import streamlit as st
    api_key = st.secrets["GEMINI_API_KEY"].strip()
except Exception:
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("❌ ไม่พบ API Key กรุณาระบุใน .streamlit/secrets.toml หรือ Environment Variables")
    exit(1)

# นำเข้าฟังก์ชันจาก rag_pipeline
try:
    from rag_pipeline import get_qa_chain
except ImportError:
    print("❌ ไม่พบโมดูล rag_pipeline กรุณารันสคริปต์นี้ในไดเรกทอรีเดียวกับโปรเจกต์")
    exit(1)

# ปิด Warning ของ Google GenAI
import warnings
warnings.filterwarnings("ignore")

def run_evaluation(dataset_file="golden_dataset.json", report_file="evaluation_report.json"):
    print("="*50)
    print("🎯 เริ่มต้นการทดสอบ RAG Evaluation (Golden Dataset)")
    print("="*50)

    # โหลดชุดคำถาม
    if not os.path.exists(dataset_file):
        print(f"❌ ไม่พบไฟล์ชุดทดสอบ {dataset_file}")
        return

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"กำลังโหลดฐานข้อมูล RAG (Vector DB)...")
    qa_chain = get_qa_chain(api_key)
    if not qa_chain:
        print("❌ ไม่พบฐานข้อมูล ChromaDB กรุณากดปุ่ม 'โหลดเอกสาร' ในหน้าเว็บแอปก่อนครับ")
        return

    results = []
    completed_ids = set()
    if os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                results = json.load(f)
                completed_ids = {r.get("question_id") for r in results if "question_id" in r}
                print(f"✅ โหลดข้อมูลเก่าที่รันไปแล้ว {len(completed_ids)} ข้อ")
        except Exception:
            pass

    total_questions = len(dataset)
    
    for i, item in enumerate(dataset, 1):
        if i in completed_ids:
            continue
            
        question = item.get("question", "")
        expected = item.get("expected_answer", "")
        
        print(f"\n[{i}/{total_questions}] คำถาม: {question}")
        print(f"  👉 เฉลยที่ตั้งไว้: {expected}")
        
        # ค้นหาคำตอบจาก AI บอท
        context_texts = []
        try:
            start_time = time.time()
            response = qa_chain.invoke({"input": question})
            ai_answer = response["answer"]
            
            # เก็บ Context ที่ดึงมาได้
            source_docs = response.get("context", [])
            for doc in source_docs:
                context_texts.append(doc.page_content)
                
            elapsed_time = time.time() - start_time
            print(f"  🤖 คำตอบจาก AI ({elapsed_time:.2f} วินาที): \n     {ai_answer}")
            
        except Exception as e:
            ai_answer = f"Error: ขออภัยเกิดข้อผิดพลาด {str(e)}"
            print(f"  ⚠️ ประมวลผลผิดพลาด: {e}")
            
        # บันทึกผลลัพธ์ลงรายงาน และเซฟทันที
        results.append({
            "question_id": i,
            "question": question,
            "expected_answer": expected,
            "ai_answer": ai_answer,
            "context_texts": context_texts,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        time.sleep(2) # หน่วงเวลาพักเล็กน้อยเพื่อหลีกเลี่ยง API Rate Limit

    # (เซฟทุกข้อแล้ว ไม่ต้องเซฟตรงนี้อีก)
        
    print("\n" + "="*50)
    print(f"✅ การทดสอบเสร็จสิ้น! บันทึกรายงานผลไปที่ไฟล์ {report_file} เรียบร้อยแล้ว")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()
