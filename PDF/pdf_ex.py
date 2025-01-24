import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
import faiss

# Set API keys and URLs from environment variables or default values
API_KEY = os.getenv("SAMBA_API_KEY", "540f8914-997e-46c6-829a-ff76f5d4d265")
API_URL = os.getenv("SAMBA_API_URL", "https://api.sambanova.ai/v1/chat/completions")

# Load SentenceTransformer embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper around SentenceTransformer for compatibility."""
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True)

def extract_text_from_pdf(pdf_path: str, chunk_size=500, overlap=100) -> list:
    """Extract text from a PDF file with chunk overlap to ensure context continuity."""
    content_chunks = []
    try:
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                text = page.get_text()
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
    """Return the custom prompt template for the LLM."""
    custom_prompt_template = """<s>[INST] <<SYS>>
You are an intelligent assistant. Provide answers strictly based on the provided context.
If the answer isn't in the context, respond with:
'This information isn't available in the provided context.'

Context:
{context}

<</SYS>>

Question:
{question}

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

    retriever = VectorStoreRetriever(vectorstore=faiss_index, search_kwargs={"k": 15})

    llm = ChatSambaNovaCloud(
        model="llama3-405b",
        max_tokens=512,
        temperature=0.1,  # Lower temperature for precision
        top_k=5,
        top_p=0.5,
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
            retrieved_docs = retriever.get_relevant_documents(user_query)
            print(f"Retrieved Documents: {retrieved_docs[:2]}")  # Debugging: Check retrieved content

            result = qa_chain.invoke({"query": user_query})
            if 'result' in result:
                print(f"Response: {result['result']}")
            else:
                print("No valid response generated.")
        except Exception as e:
            print(f"Error during query: {e}")

if __name__ == "__main__":
    main()
