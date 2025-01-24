import os
import faiss
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from nltk.tokenize import sent_tokenize

API_KEY = os.getenv("SAMBA_API_KEY", "540f8914-997e-46c6-829a-ff76f5d4d265")
API_URL = os.getenv("SAMBA_API_URL", "https://api.sambanova.ai/v1/chat/completions")

embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper around SentenceTransformer for compatibility with LangChain."""
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True)

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a DOCX file."""
    text = ""
    try:
        doc = Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text

def chunk_text(text, chunk_size=500):
    """Split text into chunks of approximately 'chunk_size' tokens."""
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def create_faiss_index(content_chunks):
    """Create a FAISS index from content chunks."""
    try:
        embeddings = SentenceTransformerEmbeddings(embedding_model)
        faiss_index = FAISS.from_texts(
            texts=content_chunks,
            embedding=embeddings,
            metadatas=[{"source": "docx", "index": i} for i in range(len(content_chunks))]
        )
        return faiss_index
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        raise

def get_custom_prompt():
    """Generate a custom prompt template."""
    custom_prompt_template = """<s>[INST] <<SYS>>
    You are a knowledgeable assistant. Use the provided context to answer the user's question accurately.
    If you cannot find the answer in the context, respond: 'Information not available.'
    
    Context:
    {context}
    <</SYS>>

    Question:
    {question}

    Answer concisely and stay relevant to the context provided.
    [/INST]"""
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def main():
    """Main function to handle user input and queries."""
    docx_path = input("Enter the path to the DOCX file: ").strip()

    try:
        docx_text = extract_text_from_docx(docx_path)
        if not docx_text.strip():
            print("No text extracted from the DOCX.")
            return
        print("DOCX Text Extracted Successfully.")
    except Exception as e:
        print(f"Failed to extract DOCX text: {e}")
        return

    content_chunks = chunk_text(docx_text)
    print(f"Extracted {len(content_chunks)} content chunks.")

    try:
        faiss_index = create_faiss_index(content_chunks)
        print("FAISS index created successfully.")
    except Exception as e:
        print(f"Failed to create FAISS index: {e}")
        return
    retriever = VectorStoreRetriever(vectorstore=faiss_index, search_kwargs={"nprobe": 5})
    
    llm = ChatSambaNovaCloud(
        model="llama3-405b",
        max_tokens=1024,
        temperature=0.5,
        top_k=3,
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
            result = qa_chain.run(user_query)
            print(f"Response: {result}")
        except Exception as e:
            print(f"Error during query: {e}")

if __name__ == "__main__":
    main()
