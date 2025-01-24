import fitz
import os
import asyncio
import aiohttp

API_KEY = os.getenv("SAMBA_API_KEY", "540f8914-997e-46c6-829a-ff76f5d4d265")
API_URL = os.getenv("SAMBA_API_URL", "https://api.sambanova.ai/v1/chat/completions")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF using PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            text += page.get_text()
    return text

def chunk_text(text: str, max_length: int = 8000) -> list:
    """Splits the text into smaller chunks."""
    words = text.split()
    chunks, current_chunk = [], []

    for word in words:
        if sum(len(w) for w in current_chunk) + len(word) < max_length:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

async def summarize_chunk(session, chunk: str) -> str:
    """Asynchronously summarize a single chunk."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-405b",
        "messages": [{"role": "user", "content": chunk}],
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_k": 1,
        "top_p": 0.01
    }

    async with session.post(API_URL, json=payload, headers=headers, timeout=30) as response:
        if response.status == 200:
            result = await response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error: {response.status} - {await response.text()}")

async def summarize_large_text(text: str) -> str:
    """Asynchronously summarizes large text by processing chunks."""
    chunks = chunk_text(text)
    summaries = []

    async with aiohttp.ClientSession() as session:
        tasks = [summarize_chunk(session, chunk) for chunk in chunks]

        for task in asyncio.as_completed(tasks):
            try:
                summary = await task
                summaries.append(summary)
            except Exception as e:
                # Log the error for debugging without printing it out
                # You can print to a log file or handle it as you wish
                # print(f"Error summarizing a chunk: {e}")
                pass

    return "\n".join(summaries)

if __name__ == "__main__":
    pdf_path = input("Enter the path to the PDF file: ").strip()

    # Extract text from PDF
    try:
        pdf_text = extract_text_from_pdf(pdf_path)
        print("PDF Text Extracted Successfully.")
    except Exception as e:
        print(f"Failed to extract PDF text: {e}")
        exit(1)

    # Summarize the text asynchronously
    try:
        summary = asyncio.run(summarize_large_text(pdf_text))
        print("Final Summary:\n", summary)
    except Exception as e:
        print(f"Failed to summarize text: {e}")
