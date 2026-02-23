import google.generativeai as genai

genai.configure(api_key="AIzaSyDntYbI6uGq81o_jW5o0MLCE6PYBmbo9Bo")
print("FULL MODEL LIST:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"MODEL: {m.name}")
