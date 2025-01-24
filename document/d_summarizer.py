import requests
import os
from docx import Document

API_KEY = os.getenv("SAMBA_API_KEY", "540f8914-997e-46c6-829a-ff76f5d4d265")
API_URL = os.getenv("SAMBA_API_URL", "https://api.sambanova.ai/v1/chat/completions")

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a DOCX file."""
    text = ""
    try:
        doc = Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""
    return text

def summarize_text(text: str) -> str:
    """Use SambaNova API to summarize the given text."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-405b",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_k": 1,
        "top_p": 0.01
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status() 
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"API Request Failed: {e}")
        return "Failed to generate a summary."

if __name__ == "__main__":
    docx_path = input("Enter the path to the DOCX file: ").strip()

    try:
        docx_text = extract_text_from_docx(docx_path)
        if not docx_text.strip():
            print("No text found in the DOCX file.")
            exit(1)
        print("DOCX Text Extracted Successfully.")
    except Exception as e:
        print(f"Error during text extraction: {e}")
        exit(1)

    try:
        summary = summarize_text(docx_text)
        print("\nSummary:\n", summary)
    except Exception as e:
        print(f"Error during summarization: {e}")
