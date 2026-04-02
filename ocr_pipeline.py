import os
import base64
import json
import re
import google.generativeai as genai

def extract_receipt_data(image_bytes: bytes, api_key: str):
    """
    ใช้ google.generativeai SDK โดยตรง เพื่อลดปัญหา Model Name Mapping ใน LangChain
    สกัดข้อมูลใบเสร็จออกมาเป็น JSON
    """
    if not api_key:
        raise ValueError("API Key is required for OCR extraction.")
        
    genai.configure(api_key=api_key)
    
    # วบรวมชื่อรุ่นที่เป็นไปได้ (Gemini API ภูมิภาคต่างๆ อาจใช้ชื่อรุ่นต่างกัน)
    # เราจะลอง 1.5 Flash เป็นหลักเพราะฟรีและเก่งรูปภาพ
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-2.0-flash-exp", # ตัวใหม่ล่าสุด (ถ้ามี)
        "gemini-pro-vision"    # รุ่นเก่า (เป็น fallback สุดท้าย)
    ]
    
    prompt = """
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
    
    # แปลงภาพ
    contents = [
        prompt,
        {"mime_type": "image/jpeg", "data": image_bytes}
    ]
    
    last_error = None
    for model_id in models_to_try:
        try:
            model = genai.GenerativeModel(model_id)
            response = model.generate_content(contents)
            
            # สกัด JSON ออกจาก Text
            text = response.text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "ไม่สามารถแปลงผลลัพธ์เป็น JSON ได้", "raw_content": text}
                
        except Exception as e:
            last_error = str(e)
            if "not found" in last_error.lower() or "404" in last_error:
                continue # ลองรุ่นถัดไป
            else:
                return {"error": f"เกิดข้อผิดพลาดในการรัน {model_id}: {last_error}"}
                
    return {"error": f"ไม่พบรุ่นโมเดลที่ใช้งานได้ (404 ทั้งหมด): {last_error}"}

def verify_receipt_rules(qa_chain, ocr_data):
    """
    ตรวจสอบข้อมูลใบเสร็จเทียบกับกฎระเบียบ (ยังคงใช้ QA Chain เพราะเป็น Text-based)
    """
    if "error" in ocr_data:
        return f"ไม่สามารถตรวจสอบคำขอได้เนื่องจาก: {ocr_data['error']}"
        
    vendor_name = ocr_data.get("vendor_name", "ไม่ระบุ")
    total_amount = ocr_data.get("total_amount", 0.0)
    
    question = (
        f"มีใบเสร็จยอดเงิน {total_amount} บาท จากร้าน '{vendor_name}' "
        f"ตามระเบียบพัสดุและงบประมาณของสถาบัน การเบิกจ่ายยอดเงินเท่านี้ต้องใช้เอกสารหลักฐานอะไรเพิ่มเติมบ้าง "
        f"และบิลนี้มีข้อมูลครบถ้วนพอจะเบิกได้หรือไม่?"
    )
    
    try:
        result = qa_chain.invoke({"input": question})
        return result["answer"]
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการตรวจสอบตามกฎระเบียบ: {str(e)}"
