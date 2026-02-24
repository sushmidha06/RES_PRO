import PyPDF2
import os

pdf_path = "mine.pdf"
if os.path.exists(pdf_path):
    print(f"File size: {os.path.getsize(pdf_path)} bytes")
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        print(f"Number of pages: {len(reader.pages)}")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            print(f"--- Page {i+1} content (first 100 chars) ---")
            print(f"'{text[:100]}'")
else:
    print("File not found")
