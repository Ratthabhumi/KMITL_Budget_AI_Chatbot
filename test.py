import traceback
try:
    from rag_pipeline import initialize_vector_db
    print("Success")
except Exception as e:
    traceback.print_exc()
