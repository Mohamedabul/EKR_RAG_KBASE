import csv
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

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper around SentenceTransformer for compatibility."""
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True)

def extract_text_from_csv(csv_path: str) -> str:
    """Extracts and formats text from a CSV file with a fallback encoding."""
    text = ""
    try:
        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                text += " ".join(row) + "\n"
    except UnicodeDecodeError:
        with open(csv_path, mode='r', encoding='ISO-8859-1') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                text += " ".join(row) + "\n"
    return text


def chunk_text(text: str, max_length: int = 24000) -> list:
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
    prompt = f"""You are an intelligent assistant. Summarize the following content:

    {chunk}

    Provide only a concise and coherent summary of the data without including additional content or interpretations."""

    payload = {
        "model": "llama3-70b",
        "messages": [{"role": "user", "content": prompt}],
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
        metadatas=[{"source": "csv", "index": i} for i in range(len(content_chunks))]
    )
    return faiss_index

def get_custom_prompt():
    custom_prompt_template = """<s>[INST] <<SYS>>
    You are an advanced information extraction and question-answering assistant designed to provide comprehensive, detailed responses 
    based strictly on the context from the uploaded CSV file. Your goal is to:

    1. Thoroughly analyze the entire context
    2. Extract ALL relevant information related to the user's query
    3. Provide a comprehensive, multi-faceted response
    4. If information is partially available, include all related details
    5. Organize the response in a clear, structured manner
    6. Be as exhaustive as possible within the context of the provided data

    Important Guidelines:
    - If the query can be answered completely or partially from the context, provide a detailed response
    - Include multiple perspectives or aspects related to the query
    - If some information is missing, clearly state which parts are covered
    - Avoid adding any external or hypothetical information
    - If no information is found, explicitly explain that no relevant information exists in the context

    Context:
    {context}

    <</SYS>>

    User Query: {question}

    Detailed Response Requirements:
    - Provide a comprehensive answer
    - Break down the response into clear sections if multiple aspects are relevant
    - Cite specific details from the context
    - If the information is insufficient, explain exactly what is missing

    [/INST]"""
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def save_summary_and_interaction_to_mongodb(csv_summary, interactions):
    """Save the CSV summary and interactions to MongoDB."""
    document = {
        "csv_summary": csv_summary,
        "interactions": interactions
    }
    collection.insert_one(document)
    print("Summary and interactions saved to MongoDB.")

async def main():
    csv_path = input("Enter the path to the CSV file: ").strip()

    try:
        csv_text = extract_text_from_csv(csv_path)
        print("CSV Text Extracted Successfully.")
        csv_summary = await summarize_large_text(csv_text)
        print("Summary of CSV:\n", csv_summary)
    except Exception as e:
        print(f"Failed to extract or summarize CSV text: {e}")
        return

    content_chunks = chunk_text(csv_text)
    try:
        faiss_index = create_faiss_index(content_chunks)
        print("FAISS index created successfully.")
    except Exception as e:
        print(f"Failed to create FAISS index: {e}")
        return

    retriever = VectorStoreRetriever(vectorstore=faiss_index, search_kwargs={"k": 5})
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

    # save_summary_and_interaction_to_mongodb(csv_summary, interactions)

if __name__ == "__main__":
    asyncio.run(main())
