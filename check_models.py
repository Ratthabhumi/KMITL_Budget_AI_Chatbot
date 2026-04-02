import google.generativeai as genai
import os
import streamlit as st

# ลองดึง key จาก environment หรือ secrets
api_key = os.environ.get("GOOGLE_API_KEY") 
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        pass

if api_key:
    genai.configure(api_key=api_key)
    print("Listing available models:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"Error listing models: {e}")
else:
    print("API Key not found.")
