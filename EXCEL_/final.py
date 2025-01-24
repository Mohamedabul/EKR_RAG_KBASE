import os
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
import faiss
from pymongo import MongoClient

API_KEY = os.getenv("SAMBA_API_KEY", "540f8914-997e-46c6-829a-ff76f5d4d265")
API_URL = os.getenv("SAMBA_API_URL", "https://api.sambanova.ai/v1/chat/completions")
MONGO_URL = "mongodb+srv://2217028:NquBh8oPPopA0Zuu@sumrag.ux9hs.mongodb.net/?retryWrites=true&w=majority&appName=SUMRAG"
DB_NAME = "DT3_EKR_BASE"
COLLECTION_NAME = "vectors"

client = MongoClient(MONGO_URL)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True)

def extract_text_from_excel(excel_path: str) -> str:
    try:
        excel_data = pd.read_excel(excel_path, sheet_name=None, engine='openpyxl')
        summary_content = ""
        for sheet_name, df in excel_data.items():
            if df.empty:
                summary_content += f"Sheet: {sheet_name} is empty.\n\n"
            else:
                first_row = df.iloc[0].to_string(index=False) if len(df) > 0 else "No data rows found"
                summary_content += (
                    f"Sheet: {sheet_name}\n"
                    f"Columns: {', '.join(df.columns)}\n"
                    f"Total Rows: {len(df)}\n"
                    f"First Row Data:\n{first_row}\n\n"
                )
        return summary_content if summary_content else "No sheets found in the Excel file."
    except Exception as e:
        raise Exception(f"Failed to extract data from Excel: {e}")

def summarize_text(text: str) -> str:
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
    response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

def create_faiss_index(content_chunks):
    embeddings = SentenceTransformerEmbeddings(embedding_model)
    faiss_index = FAISS.from_texts(
        texts=content_chunks,
        embedding=embeddings,
        metadatas=[{"source": f"Sheet_{i}", "index": i} for i in range(len(content_chunks))]
    )
    return faiss_index

def get_custom_prompt():
    custom_prompt_template = """<s>[INST] <<SYS>>
    You are an intelligent assistant designed to extract and provide detailed, accurate answers based on the contents of an document.
    Use the following context, extracted from the file, to answer the user's questions. If the question is related to specific data in the document, locate and include all relevant information.

    Excel Context:
    {context}

    <</SYS>>

    Please answer the following question:
    {question}

    Ensure your response is based on precise data from the Excel file and covers all details available in the context. If you cannot find an answer within the data, respond with:
    'This information isn't available in the provided data.'
    [/INST]"""
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])


def save_to_mongodb(excel_summary, queries_responses):
    document = {
        "excel_summary": excel_summary,
        "queries_responses": queries_responses
    }
    collection.insert_one(document)
    print("Data successfully saved to MongoDB.")

def main():
    file_path = input("Enter the path to the Excel file (.xls or .xlsx): ").strip()
    file_type = file_path.split(".")[-1].lower()
    
    if file_type not in ["xls", "xlsx"]:
        raise ValueError("Unsupported file type! Please provide an Excel file (.xls or .xlsx).")

    try:
        excel_summary = extract_text_from_excel(file_path)
        # print("Excel Data Extracted Successfully.\n", excel_summary)

        summary = summarize_text(excel_summary)
        print("\nSummary:\n", summary)

        content_chunks = [excel_summary]
        faiss_index = create_faiss_index(content_chunks)
        retriever = VectorStoreRetriever(vectorstore=faiss_index)

        llm = ChatSambaNovaCloud(
            model="llama3-405b",
            max_tokens=1024,
            temperature=0.7,
            top_k=1,
            top_p=0.01,
            request_timeout=30
        )
        
        custom_prompt = get_custom_prompt()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": custom_prompt, "document_variable_name": "context"}
        )

        queries_responses = []

        while True:
            user_query = input("Ask a question (or type 'exit'): ").strip()
            if user_query.lower() == 'exit':
                # save_to_mongodb(summary, queries_responses)
                print("Thank you! Exiting.")
                break

            try:
                result = qa_chain.invoke({"query": user_query})
                response = result['result']
                print(f"Response: {response}")
                queries_responses.append({"query": user_query, "response": response})
            except Exception as e:
                print(f"Error during query: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
