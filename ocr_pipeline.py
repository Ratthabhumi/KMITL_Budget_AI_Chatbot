import os
import base64
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def extract_receipt_data(image_bytes: bytes, api_key: str):
    """
    รับรูปภาพใบเสร็จเป็นก้อนไบต์ (bytes) ส่งเข้าไปวิเคราะห์ผ่าน Gemini 1.5 Flash 
    เพื่อดึงข้อมูลให้ออกมาในรูปแบบ JSON ตามโครงสร้างที่กำหนด
    """
    if not api_key:
        raise ValueError("API Key is required for OCR extraction.")
        
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # แปลงไบต์ภาพเป็น Base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # กำหนด Prompt สำหรับการทำ OCR และสกัดข้อมูลลง JSON Schema
    sys_prompt = """
    คุณคือผู้เชี่ยวชาญด้านบัญชีและการตรวจสอบเอกสารทางการเงิน
    หน้าที่ของคุณคืออ่านรูปภาพใบเสร็จรับเงิน หรือใบกำกับภาษีของประเทศไทย
    และดึงข้อมูลที่สำคัญออกมาในรูปแบบ JSON เท่านั้น ห้ามตอบนอกเหนือจากรูปแบบที่กำหนด

    รูปแบบ JSON ที่ต้องการ:
    {
        "vendor_name": "ชื่อร้านค้า หรือ บริษัทที่ออกใบเสร็จ (ถ้าหาไม่เจอให้ใส่ null)",
        "tax_id": "เลขประจำตัวผู้เสียภาษี (ถ้าเป็นใบกำกับภาษี, ถ้าหาไม่เจอให้ใส่ null)",
        "transaction_date": "วันที่ออกใบเสร็จ (ถ้าหาไม่เจอให้ใส่ null)",
        "total_amount": ตัวเลขยอดรวมสุทธิ (เป็น float ตัวเลขเท่านั้น ไม่มีลูกน้ำ),
        "items": [
            {
                "description": "ชื่อรายการสินค้า",
                "price": ตัวเลขราคาสินค้านั้นๆ (float)
            }
        ]
    }
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": sys_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    )

    # ลองแต่ละโมเดลตามลำดับ (Gemma API Key มักจะมีสิทธิ์ใช้รุ่นเหล่านี้)
    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-latest"]
    last_error = None
    for model_name in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
            response = llm.invoke([message])
            content = response.content

            # แกะ JSON ออกจากข้อความ (เผื่อโมเดลตอบติด Markdown ```json มาด้วย)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "ไม่สามารถแปลงผลลัพธ์เป็น JSON ได้", "raw_content": content}
        except Exception as e:
            last_error = e
            # ถ้าเป็น quota/rate-limit error ให้ลองโมเดลถัดไป ไม่ใช่หยุด
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e) or "quota" in str(e).lower():
                continue
            # error อื่นๆ หยุดทันที
            return {"error": str(e)}

    return {"error": f"โมเดลทุกตัวถูก rate-limit: {str(last_error)}"}

def verify_receipt_rules(qa_chain, ocr_data):
    """
    นำข้อมูลที่สกัดได้จากใบเสร็จ ไปตรวจสอบความถูกต้องเทียบกับกฎระเบียบ 
    โดยใช้ RAG (QA Chain) ดึงกฎและให้ AI สรุปผล
    """
    if "error" in ocr_data:
        return f"ไม่สามารถตรวจสอบคำขอได้เนื่องจาก: {ocr_data['error']}"
        
    vendor_name = ocr_data.get("vendor_name", "ไม่ระบุ")
    total_amount = ocr_data.get("total_amount", 0.0)
    
    # ตั้งคำถามให้ RAG สรุปผลตามระเบียบ
    question = (
        f"มีใบเสร็จยอดเงิน {total_amount} บาท จากร้าน '{vendor_name}' "
        f"ตามระเบียบพัสดุและงบประมาณของสถาบัน การเบิกจ่ายยอดเงินเท่านี้ต้องใช้เอกสารหลักฐานอะไรเพิ่มเติมบ้าง "
        f"และบิลนี้มีข้อมูลครบถ้วนพอจะเบิกได้หรือไม่?"
    )
    
    try:
        # ใช้ qa_chain ตัวหลักของระบบประมวลผลดึง Policy มาตอบ
        response = qa_chain.invoke({"input": question})
        return response["answer"]
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการตรวจสอบตามกฎระเบียบ: {str(e)}"
