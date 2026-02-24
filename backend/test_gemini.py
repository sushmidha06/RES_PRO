import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def find_working_model(api_key):
    genai.configure(api_key=api_key)
    print("Listing models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Testing model: {m.name}")
            try:
                model = genai.GenerativeModel(m.name)
                response = model.generate_content("Hi")
                print(f"Success with: {m.name}")
                return m.name
            except Exception as e:
                print(f"Failed with {m.name}: {e}")
    return None

if __name__ == "__main__":
    key = os.getenv("GEMINI_API_KEY")
    working_model = find_working_model(key)
    print(f"FINAL_WORKING_MODEL: {working_model}")
