# 🎓 KMITL Budget AI Chatbot

**AI Chatbot สำหรับตอบคำถามด้านการเบิกจ่ายงบประมาณและการจัดซื้อจัดจ้างของ สจล.**

> Course: Computer Engineering Project Preparation (CEPP) — 01276390  
> King Mongkut's Institute of Technology Ladkrabang (KMITL)

---

## 🤔 ระบบนี้ทำอะไร?

แทนที่จะให้บุคลากรค้นหาคำตอบจากเอกสาร PDF หลายสิบไฟล์ด้วยตัวเอง ระบบนี้จะ **อ่านเอกสาร PDF ทั้งหมดและตอบคำถามโดยอ้างอิงจากข้อมูลจริง** พร้อมระบุว่าข้อมูลมาจากหน้าไหนของเอกสารไหน

## 🏗️ สถาปัตยกรรมระบบ (RAG Architecture)

```
📄 PDF ระเบียบการ (Docs/)
    ↓  PyMuPDF (รองรับภาษาไทย)
📝 ตัดแบ่งเป็น Chunks (1000 ตัวอักษร)
    ↓  HuggingFace Embeddings (Local — ฟรี ไม่ติดโควตา)
🗄️ ChromaDB (Vector Database เก็บในเครื่อง)
    ↓  ผู้ใช้ถามคำถาม → ดึง 12 ชิ้นที่เกี่ยวข้องที่สุด
🤖 Gemma 3 4B via Google AI API (ฟรี 14,400 req/วัน)
    ↓
💬 คำตอบภาษาไทย + แสดงแหล่งอ้างอิง PDF ต้นฉบับ
```

---

## ⚙️ Technology Stack

| Component | Technology |
|---|---|
| UI Framework | [Streamlit](https://streamlit.io/) |
| PDF Loader | PyMuPDF (รองรับภาษาไทยได้ดี) |
| Text Embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Local) |
| Vector Database | [ChromaDB](https://www.trychroma.com/) |
| LLM | `gemma-3-4b-it` via Google AI API |
| RAG Framework | [LangChain](https://www.langchain.com/) |

---

## 🚀 วิธีติดตั้งและรัน

### 1. Clone และติดตั้ง

```bash
git clone https://github.com/Ratthabhumi/CEIPP.git
cd CEIPP
pip install -r requirements.txt
```

### 2. ตั้งค่า API Key

```bash
# คัดลอก template
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# แล้วเปิดไฟล์ .streamlit/secrets.toml และใส่ API Key
# รับ Key ฟรีได้ที่: https://aistudio.google.com/
```

### 3. เพิ่มเอกสาร PDF

วาง PDF ระเบียบการทั้งหมดลงในโฟลเดอร์ `Docs/`

### 4. รันระบบ

```bash
streamlit run app.py
```

> กดปุ่ม **"โหลดเอกสารจากโฟลเดอร์ Docs"** ครั้งแรกเพื่อสร้าง Vector Database (ทำแค่ครั้งเดียว)

---

## 📊 ผลการประเมินความแม่นยำ

ใช้ระบบ **LLM-as-a-Judge** (คล้าย RAGAS) ให้ AI ประเมินตัวเองใน 3 ด้าน:

| เมตริก | คะแนน | ความหมาย |
|---|---|---|
| **Faithfulness** | 4.7 / 5 | ไม่มั่วข้อมูล อ้างอิงจากเอกสารจริง |
| **Answer Relevance** | 4.3 / 5 | ตอบตรงประเด็น |
| **Context Precision** | 4.0 / 5 | ดึงเอกสารถูกมาให้ AI |
| **ความเร็วตอบ** | ~4.3 วินาที | ใช้งานได้จริง |

### รันการทดสอบ

```bash
python evaluate.py       # สร้าง evaluation_report.json
python ai_evaluator.py   # ให้ AI ให้คะแนนผลลัพธ์
```

---

## 📁 โครงสร้างโปรเจกต์

```
CEIPP/
├── app.py                  # Streamlit UI หลัก
├── rag_pipeline.py         # RAG Logic (Embedding + Retrieval + LLM)
├── build_db.py             # Script สร้าง Vector DB ครั้งแรก
├── evaluate.py             # Automated Testing
├── ai_evaluator.py         # LLM-as-a-Judge Scoring
├── golden_dataset.json     # ชุดข้อสอบสำหรับทดสอบ
├── requirements.txt        # Python Dependencies
├── Docs/                   # ⚠️ วาง PDF ระเบียบการที่นี่ (ไม่ขึ้น GitHub)
├── chroma_db/              # ⚠️ Vector DB (auto-generated, ไม่ขึ้น GitHub)
└── .streamlit/
    ├── secrets.toml        # ⚠️ API Key (ไม่ขึ้น GitHub!)
    └── secrets.toml.example # Template วิธีตั้งค่า
```

---

## 🔮 แผนพัฒนาต่อ

- [ ] เพิ่ม PDF ระเบียบการให้ครอบคลุม
- [ ] Dashboard แสดงผลคะแนน Evaluation
- [ ] เพิ่มชุดข้อสอบใน Golden Dataset
- [ ] รองรับการค้นหาแบบ Hybrid Search (BM25 + Vector)