import fitz
import os
import asyncio
import aiohttp
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
import faiss

API_KEY = os.getenv("SAMBA_API_KEY", "540f8914-997e-46c6-829a-ff76f5d4d265")
API_URL = os.getenv("SAMBA_API_URL", "https://api.sambanova.ai/v1/chat/completions")

client = MongoClient("mongodb+srv://2217028:NquBh8oPPopA0Zuu@sumrag.ux9hs.mongodb.net/?retryWrites=true&w=majority&appName=SUMRAG")
db = client['DT3_EKR_BASE']
collection = db['vectors']


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper around SentenceTransformer for compatibility."""
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True)

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True, device=device)
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF using PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            text += page.get_text()
    return text

def chunk_text(text: str, max_length: int = 16000) -> list:
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
        "model": "llama3-70b",
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
                pass

    return "\n".join(summaries)

def create_faiss_index(content_chunks): 
    """Create a FAISS index from content chunks."""
    embeddings = SentenceTransformerEmbeddings(embedding_model)
    faiss_index = FAISS.from_texts(
        texts=content_chunks,
        embedding=embeddings,
        metadatas=[{"source": "pdf", "index": i} for i in range(len(content_chunks))]
    )
    return faiss_index

def get_custom_prompt():
    custom_prompt_template = """<s>[INST] <<SYS>>
    You are an intelligent assistant designed to extract and provide comprehensive, detailed answers based on the content of a document.
    Use the following context, extracted from the file, to answer the user's questions. When responding, locate and include all relevant information related to the question from the document.

    Context:
    {context}

    <</SYS>>

    Please answer the following question:
    {question}

    Ensure your response includes all available and relevant content from the document to fully address the user's query. If the exact answer is not found within the data, respond with:
    'This information isn't available in the provided data.'
    [/INST]"""
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])


def save_summary_and_interaction_to_mongodb(pdf_summary, interactions):
    """Save the PDF summary and interactions to MongoDB."""
    document = {
        "pdf_summary": pdf_summary,
        "interactions": interactions
    }
    collection.insert_one(document)
    print("Summary and interactions saved to MongoDB.")

async def main():
    pdf_path = input("Enter the path to the PDF file: ").strip()

    try:
        pdf_text = extract_text_from_pdf(pdf_path)
        print("PDF Text Extracted Successfully.")
        pdf_summary = await summarize_large_text(pdf_text)
        print("Summary of PDF:\n", pdf_summary)
    except Exception as e:
        print(f"Failed to extract or summarize PDF text: {e}")
        return

    content_chunks = chunk_text(pdf_text)  
    try:
        faiss_index = create_faiss_index(content_chunks)
        print("FAISS index created successfully.")
    except Exception as e:
        print(f"Failed to create FAISS index: {e}")
        return

    retriever = VectorStoreRetriever(vectorstore=faiss_index, search_kwargs={"k": 15})
    llm = ChatSambaNovaCloud(
        model="llama3-70b",
        max_tokens=512,
        temperature=0.5,
        top_k=1,
        top_p=0.8,
        request_timeout=30
    )
    custom_prompt = get_custom_prompt()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt, "document_variable_name": "context"}
    )

    interactions = []
    while True:
        user_query = input("Ask a question (or type 'exit'): ").strip()
        if user_query.lower() == 'exit':
            print("Thank you for using the assistant!")
            break

        try:
            result = qa_chain.invoke({"query": user_query})
            response = result['result']
            print(f"Response: {response}")
            interactions.append({"query": user_query, "response": response})
        except Exception as e:
            print(f"Error during query: {e}")

    save_summary_and_interaction_to_mongodb(pdf_summary, interactions)

if __name__ == "__main__":
    asyncio.run(main())
