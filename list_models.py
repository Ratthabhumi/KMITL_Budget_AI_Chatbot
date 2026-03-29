import google.generativeai as genai

genai.configure(api_key="AIzaSyDsXWfTx5iMZQZg2AMwonbDXL2N-4_L9z0")

try:
    models = genai.list_models()
    for m in models:
        if 'embedContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error: {e}")
