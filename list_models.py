import google.generativeai as genai

import os

api_key = os.environ.get("GEMINI_API_KEY")

try:
    import streamlit as st
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    print("⚠️ ไม่พบข้อมูล API Key กรุณาระบุใน secrets.toml")
    exit()

genai.configure(api_key=api_key)

try:
    models = genai.list_models()
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error: {e}")
