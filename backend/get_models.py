import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
with open('models_list.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(models))
print(f"Wrote {len(models)} models to models_list.txt")
