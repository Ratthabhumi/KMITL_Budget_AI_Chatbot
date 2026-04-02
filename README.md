# 🎓 KMITL Budget AI Chatbot
**AI Chatbot สำหรับตอบคำถามด้านการเบิกจ่ายงบประมาณและการจัดซื้อจัดจ้างของ สจล.**

**Course:** Computer Engineering Project Preparation (CEPP) — 01276390  
**King Mongkut's Institute of Technology Ladkrabang (KMITL)**

---

## 🤔 ระบบนี้ทำอะไร?
แทนที่จะให้บุคลากรค้นหาคำตอบจากเอกสาร PDF หลายสิบไฟล์ด้วยตัวเอง ควานหาตามหน้าต่างๆ ให้เสียเวลา ระบบนี้จะทำหน้าที่ **อ่านเอกสาร PDF ทั้งหมดและตอบคำถามโดยอ้างอิงจากข้อมูลจริง** พร้อมระบุอย่างชัดเจนว่าข้อมูลอ้างอิงมาจาก โฟลเดอร์ไหน ไฟล์อะไร และพิกัดหน้าที่เท่าไหร่

---

## 🏗️ สถาปัตยกรรมระบบ (RAG Architecture)

```text
📄 PDF ระเบียบการ (Docs/)
    ↓  (ใช้ PyMuPDF: ตัดและอ่านภาษาไทยได้อย่างแม่นยำ)
📝 ตัดแบ่งเป็น Chunks (1000 ตัวอักษร)
    ↓  (HuggingFace Embeddings (Local) — ไม่ต้องใช้ API, ไม่ติดโควตา)
🗄️ ChromaDB (Vector Database เก็บในเครื่อง)
    ↓  (ผู้ใช้ถามคำถาม → ดึง 12 เนื้อหาที่เกี่ยวข้องและใกล้เคียงที่สุดออกมา)
🤖 Gemma 3 4B via Google AI API (ฟรีปริมาณสูงถึง 14,400 req/วัน)
    ↓
💬 คำตอบภาษาไทยที่สรุปมาให้แล้ว + แสดงแหล่งอ้างอิง PDF ต้นฉบับ
```

---

## ⚙️ Technology Stack
| Component | Technology |
| --- | --- |
| **UI Framework** | Streamlit |
| **PDF Loader** | PyMuPDF |
| **Text Embedding** | `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2) |
| **Vector Database** | ChromaDB (`langchain-chroma`) |
| **LLM** | gemma-3-4b-it (via Google AI API) |
| **RAG Framework** | LangChain |

---

## 🚀 วิธีติดตั้งและรัน
> **ข้อกำหนดเบื้องต้น (Prerequisites):** ใช้งานได้ดีที่สุดบนโปรแกรม **Python เวอร์ชัน 3.9 - 3.11**

**1. Clone และติดตั้ง Dependencies**
```bash
git clone https://github.com/Ratthabhumi/CEIPP.git
cd CEIPP
pip install -r requirements.txt
```

**2. ตั้งค่า API Key**
```bash
# คัดลอก template ของ secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
> เปิดไฟล์ `.streamlit/secrets.toml` และนำ Google Gemini API Key มาใส่  
> *(สามารถขอ Key ฟรีได้ที่: [Google AI Studio](https://aistudio.google.com/))*

**3. จัดเตรียมโฟลเดอร์เอกสาร PDF**
ลากไฟล์ PDF เอกสารระเบียบการทั้งหมดไปวางในโฟลเดอร์ `Docs/`

**4. เตรียมฐานข้อมูลและเปิดใช้งานหน้าเว็บแอปพลิเคชัน**
```bash
streamlit run app.py
```
> **หมายเหตุเกี่ยวกับการดึงข้อมูล (Ingestion):** 
> 1. ในการรันหน้าเว็บครั้งแรก ต้องกดปุ่ม **"โหลดเอกสารจากโฟลเดอร์ Docs"** ซ้ายมือบนหน้าจอ เพื่อให้ระบบสกัดและสร้าง Vector DB (หลังทำไปแล้วครั้งแรก ไม่ต้องกดทำอีก)
> 2. การสร้างฐานข้อมูลครั้งแรก ระบบจะทำการดาวน์โหลดโมเดล Embedding แบบ Local มาไว้ในเครื่อง (ประมาน 400MB) และใช้เวลาอ่าน PDF สักพักตามความเร็ว CPU ของเครื่อง
> 3. 💡 **ทางเลือก:** หากไม่ต้องการกดเมนูผ่านหน้าเว็บ คุณสามารถสั่งสร้างฐานข้อมูลล่วงหน้าด้วยคำสั่ง `python build_db.py` ผ่าน Terminal แทนได้เช่นกัน!

---

## 📊 ผลการประเมินความแม่นยำ (Evaluation)
โปรเจกต์นี้มีระบบตรวจสอบและประเมินผลลัพธ์ของ AI ในการตอบและค้นหาอย่างครบถ้วน:

### 1. การประเมินผลลัพธ์คำตอบ (LLM-as-a-Judge)
ใช้ AI ทำหน้าที่เสมือนเป็นกรรมการ (คล้ายเฟรมเวิร์กของ Ragas) ช่วยตรวจสอบคำตอบใน 3 ด้าน ดังนี้:
- **Faithfulness (4.7 / 5):** ไม่มั่วข้อมูล (No Hallucination) ยึดตามเอกสารจริง
- **Answer Relevance (4.3 / 5):** คำตอบตรงประเด็นที่ถาม
- **Context Precision (4.0 / 5):** ดึงเนื้อหาขึ้นมาได้มีความถูกต้อง
- **ความเร็วตอบสนอง:** ~4.3 วินาที (รวดเร็วเพียงพอต่อการใช้งานจริง)

### 2. การประเมินประสิทธิภาพการค้นหา (Retrieval Evaluation)
แยกระบบประเมินวัดความสามารถของ Embedding และ Vector DB ว่าทำหน้าที่ดึงข้อมูลที่เกี่ยวข้องมาได้แม่นยำเพียงใด:
- **Recall@K:** มีคำตอบอยู่ในชุดข้อมูลที่ดึงมาให้ AI มากสุดแค่ไหน
- **Precision@K:** สัดส่วนเปอร์เซ็นต์เอกสารที่เกี่ยวข้องกันจริง ในเอกสารทั้งหมดที่หยิบมา
- **MRR (Mean Reciprocal Rank):** เอกสารที่มีประโยชน์ถูกจัดให้อยู่อันดับที่ดีหรือไม่

### 3. โครงสร้างชุดข้อสอบคู่มือ (Golden Dataset)
คำถามและคำตอบที่ถูกต้องเพื่อใช้ทดสอบกับระบบประเมิน จะถูกเก็บอยู่ในไฟล์ `golden_dataset.json` โดยมีหน้าตาโครงสร้าง Array JSON ดังนี้:
```json
[
  {
    "question": "ผู้ถือหุ้นรายใหญ่ตามระเบียบกระทรวงการคลังคือใคร?",
    "expected_answer": "หมายความถึงผู้ถือหุ้นเกินกว่าร้อยละ 25 ของทุน..."
  }
]
```

### ชุดคำสั่งรันระบบประเมินผลอัตโนมัติ
```bash
python retrieval_eval.py  # ทดสอบเฉพาะระบบค้นหาไฟล์ (จะได้ไฟล์ retrieval_eval_report.json)
python evaluate.py        # รันถามตอบและเปรียบเทียบคำตอบ (จะได้ไฟล์ evaluation_report.json)
python ai_evaluator.py    # ให้โมเดล AI ให้คะแนนผลลัพธ์ว่าตอบได้ดีแค่ไหน
```

---

## 📁 โครงสร้างโปรเจกต์
```text
CEIPP/
├── app.py                  # Streamlit UI หลัก
├── rag_pipeline.py         # RAG Logic (ประกอบร่าง Embedding + Retrieval + LLM และปรับแต่ง Prompt)
├── build_db.py             # Script สร้างฐานเวกเตอร์ (Vector DB)
├── evaluate.py             # Automated Testing สำหรับรันและสร้าง Log ประจำเซสชัน
├── ai_evaluator.py         # Script LLM-as-a-Judge สำหรับรันประเมินคะแนน
├── retrieval_eval.py       # Script วัดประสิทธิภาพการค้นหาเอกสาร (Recall, MRR) เชิงลึก
├── golden_dataset.json     # ชุดข้อสอบจำลองและคำตอบจากกฎระเบียบของ สจล.
├── requirements.txt        # ไฟล์กำหนด Dependencies แพลตฟอร์มต่างๆ
├── Docs/                   # ⚠️ โฟลเดอร์ที่เก็บไฟล์ PDF เล่มระเบียบการต่างๆ (ห้ามนำขึ้น GitHub)
├── chroma_db/              # ⚠️ โฟลเดอร์ฐานข้อมูล Vector ส่วนตัวในเครื่อง (ห้ามนำขึ้น GitHub)
└── .streamlit/
    ├── secrets.toml        # ⚠️ ข้อมูลรหัส API จำกัดสิทธิ์ (ห้ามนำขึ้น GitHub เด็ดขาด!)
    └── secrets.toml.example # แบบฟอร์มจำลองสำหรับให้ตั้งค่า API เอง
```

---

## 🔮 แผนการพัฒนาในอนาคต (Roadmap)
- [ ] เพิ่มไฟล์ PDF ระเบียบการให้ครอบคลุมการรวบรวมข้อมูลทุกกรมกอง
- [ ] พัฒนา Dashboard แสดงผลคะแนน Evaluation แยกหน้าแบบกราฟฟิกสวยงามสำหรับผู้ดูแล (Admin)
- [ ] ขยายชุดคำถามใน Golden Dataset ให้ทดสอบได้รัดกุมมากขึ้น
- [ ] เพิ่มฟีเจอร์ Hybrid Search (BM25 + Vector) เพื่อความแม่นยำสูงสุดในศัพท์ที่เฉพาะทาง