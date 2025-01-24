from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsClusteringFilter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from concurrent.futures import ThreadPoolExecutor
import os
import csv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate


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

def extract_text(file):
    file_extension = os.path.splitext(file)[1].lower()
    
    if file_extension == '.pdf':
        loader = PyPDFLoader(file)
        documents = loader.load()
    elif file_extension == '.csv':
        text = extract_text_from_csv(file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap = 200)
        documents = splitter.create_documents([text])
    else:
        raise ValueError("Unsupported file format. Please use PDF or CSV files.")
    
    return documents


def filter_documents(texts, embedding_model):
    embeddings_filter = EmbeddingsClusteringFilter(
        embeddings=embedding_model,
        num_clusters=max(1,min(len(texts) // 2, 8)),  
        num_closest=max(1,min(len(texts) // 4, 3)),    
        threshold=0.85                          
    )
    filtered_texts = embeddings_filter.transform_documents(texts)
    print("\nClustered Data:")
    for i, doc in enumerate(filtered_texts, 1):
        print(f"\nCluster {i}:")
        print(doc.page_content)
    return filtered_texts



def summarize_document(file, llm, embedding_model):
    texts = extract_text(file)
    filtered_docs = filter_documents(texts, embedding_model)
    
    prompt_template = """
    Please provide a comprehensive summary of the following document. Focus on:
    1. A detailed summary of the document
    2. Main themes and key points
    3. Important findings or conclusions
    4. Significant data or statistics if present
    5. Key recommendations or actions

    Document content:
    {text}

    Summary:
    """
    
    chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=PromptTemplate(template=prompt_template, input_variables=["text"])
    )
    
    summary = chain.invoke(filtered_docs)
    print("\nComprehensive Summary:")
    print(summary['output_text'])
    return summary




embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatSambaNovaCloud(
        model="llama3-70b",
        temperature=0.6,
        max_tokens = 4000
    )

if __name__ == "__main__":
    print("Starting document summarization...")
    print("Enter the path to your file: ", end='', flush=True)
    file_path = input()
    print(f"Processing file: {file_path}")
    
    if os.path.exists(file_path):
        summarize_document(file_path, llm, embedding_model)
    else:
        print("File not found. Please check the file path and try again.")


