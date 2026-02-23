import google.generativeai as genai
import os

genai.configure(api_key='AIzaSyDntYbI6uGq81o_jW5o0MLCE6PYBmbo9Bo')
models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
with open('models_list.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(models))
print(f"Wrote {len(models)} models to models_list.txt")
