import os
import pandas as pd
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

    def embed(self, texts):
        """Embeds a list of texts."""
        return self.embed_documents(texts)

def extract_text_from_csv(csv_path: str) -> list:
    """Extract text from a CSV file and chunk it appropriately."""
    try:
        df = pd.read_csv(csv_path)
        content_chunks = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
        
        content_chunks = [' '.join(content_chunks[i:i + 5]) for i in range(0, len(content_chunks), 5)]  
    except Exception as e:
        print(f"Error extracting CSV: {e}")
        return []
    return content_chunks

def create_faiss_index(content_chunks):
    """Create a FAISS index from content chunks."""
    try:
        embeddings = SentenceTransformerEmbeddings(embedding_model)
        faiss_index = FAISS.from_texts(
            texts=content_chunks,
            embedding=embeddings,
            metadatas=[{"source": "csv", "index": i} for i in range(len(content_chunks))]
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
    You are an intelligent assistant that uses the following context to answer the user's questions. 
    If the answer is not found, respond with: "This information isn't available in the provided context."

    Context:
    {context}

    Please answer the following question:
    {question}
    <</SYS>>[/INST]"""
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def main():
    """Main function to handle user input and queries."""
    csv_path = input("Enter the path to the CSV file: ")

    try:
        content_chunks = extract_text_from_csv(csv_path)
        if not content_chunks:
            print("No content extracted from the CSV.")
            return
        print(f"Extracted {len(content_chunks)} content chunks from the CSV.")
        print(f"Sample chunk: {content_chunks[0][:100]}")
    except Exception as e:
        print(f"Failed to extract CSV content: {e}")
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

    retriever = VectorStoreRetriever(vectorstore=faiss_index)

    llm = ChatSambaNovaCloud(
        model="llama3-405b",
        max_tokens=150,
        temperature=0.5,
        top_k=50,
        top_p=0.9,
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
