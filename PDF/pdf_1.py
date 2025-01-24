import os
import fitz  
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

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper around SentenceTransformer for compatibility."""
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True)

def extract_text_from_pdf(pdf_path: str, chunk_size=2000, overlap=1000) -> list:
    """Extract text from a PDF file with chunk overlap to ensure context continuity."""
    content_chunks = []
    try:
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                text = page.get_text()
                # Use chunking with overlap for better context retention
                for i in range(0, len(text), chunk_size - overlap):
                    content_chunks.append(text[i:i + chunk_size])
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return content_chunks

def create_faiss_index(content_chunks):
    """Create a FAISS index from content chunks."""
    try:
        embeddings = SentenceTransformerEmbeddings(embedding_model)
        faiss_index = FAISS.from_texts(
            texts=content_chunks,
            embedding=embeddings,
            metadatas=[{"source": "pdf", "index": i} for i in range(len(content_chunks))]
        )
        return faiss_index
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        raise

def save_content_and_vectors(content_chunks, faiss_index):
    """Save extracted content and FAISS index."""
    content_folder = 'extracted_content'
    index_folder = 'faiss_index'

    os.makedirs(content_folder, exist_ok=True)
    content_file_path = os.path.join(content_folder, "extracted_content.txt")
    with open(content_file_path, 'w', encoding='utf-8') as f:
        for chunk in content_chunks:
            f.write(chunk + "\n\n")
    print(f"Extracted content saved to {content_file_path}.")

    os.makedirs(index_folder, exist_ok=True)
    index_file_path = os.path.join(index_folder, "index.faiss")
    faiss.write_index(faiss_index.index, index_file_path)
    print(f"FAISS index saved to {index_file_path}.")

def get_custom_prompt():
    custom_prompt_template = """<s>[INST] <<SYS>>
    You are an intelligent assistant designed to provide information based strictly on retrieved context.
    Use only the provided context to answer the user's question. Avoid guessing or hallucinating any information.

    Context:
    {context}

    <</SYS>>

    Please answer the following question:
    {question}

    Ensure your response is concise, factual, and directly related to the provided context. If the answer isn't in the context, reply with:
    'This information isn't available in the provided context.'

    [/INST]"""
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def main():
    """Main function to handle user input and queries."""
    pdf_path = input("Enter the path to the PDF file: ")

    try:
        content_chunks = extract_text_from_pdf(pdf_path)
        if not content_chunks:
            print("No text extracted from the PDF.")
            return
        print(f"Extracted {len(content_chunks)} content chunks from the PDF.")
        print(f"Sample chunk: {content_chunks[0][:100]}")
    except Exception as e:
        print(f"Failed to extract PDF text: {e}")
        return

    try:
        faiss_index = create_faiss_index(content_chunks)
        print("FAISS index created successfully.")
    except Exception as e:
        print(f"Failed to create FAISS index: {e}")
        return

    try:
        save_content_and_vectors(content_chunks, faiss_index)
    except Exception as e:
        print(f"Error saving extracted content and FAISS index: {e}")
        return

    retriever = VectorStoreRetriever(vectorstore=faiss_index, search_kwargs={"k": 10})  # Retrieve top 15 results

    llm = ChatSambaNovaCloud(
        model="llama3-405b",
        max_tokens=1024,  
        temperature=0.2,  # Lower temperature for more deterministic answers
        top_k=1,
        top_p=0.8,  # Slightly stricter output diversity
        request_timeout=30
    )

    custom_prompt = get_custom_prompt()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt, "document_variable_name": "context"}
    )

    while True:
        user_query = input("Ask a question (or type 'exit'): ").strip()
        if user_query.lower() == 'exit':
            print("Thank you!")
            break

        try:
            result = qa_chain.invoke({"query": user_query})
            print(f"Response: {result['result']}")
        except Exception as e:
            print(f"Error during query: {e}")

if __name__ == "__main__":
    main()