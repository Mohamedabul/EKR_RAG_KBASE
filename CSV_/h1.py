import csv
import os
import asyncio
import aiohttp
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import base64
import io
import numpy as np

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings

# Configuration and API Keys
API_KEY = os.getenv("SAMBA_API_KEY", "540f8914-997e-46c6-829a-ff76f5d4d265")
API_URL = os.getenv("SAMBA_API_URL", "https://api.sambanova.ai/v1/chat/completions")

# MongoDB Connection
client = MongoClient("mongodb+srv://2217028:NquBh8oPPopA0Zuu@sumrag.ux9hs.mongodb.net/?retryWrites=true&w=majority&appName=SUMRAG")
db = client['DT3_EKR_BASE']
collection = db['vectors']

# Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper around SentenceTransformer for compatibility."""
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True)

class VisualizationGenerator:
    def __init__(self, dataframe):
        self.df = dataframe

    def generate_visualization(self, query):
        """
        Dynamically generate visualizations based on the query and data characteristics
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        def default_visualization():
            fig = px.bar(self.df, x=categorical_cols[0] if categorical_cols else None, 
                         y=numeric_cols[0] if numeric_cols else None,
                         title='Default Data Overview')
            return self._format_plot(fig)

        # Specific visualization generators
        def line_chart():
            if len(numeric_cols) >= 2:
                fig = px.line(self.df, x=numeric_cols[0], y=numeric_cols[1], 
                               title='Line Chart Visualization')
                return self._format_plot(fig)
            return None

        def bar_chart():
            if categorical_cols and numeric_cols:
                fig = px.bar(self.df, x=categorical_cols[0], y=numeric_cols[0], 
                             title='Comparative Bar Chart')
                return self._format_plot(fig)
            return None

        def pie_chart():
            if categorical_cols and numeric_cols:
                fig = px.pie(self.df, names=categorical_cols[0], values=numeric_cols[0], 
                             title='Proportion Pie Chart')
                return self._format_plot(fig)
            return None

        def area_chart():
            if len(numeric_cols) >= 2:
                fig = px.area(self.df, x=numeric_cols[0], y=numeric_cols[1], 
                              title='Cumulative Area Chart')
                return self._format_plot(fig)
            return None

        def donut_chart():
            if categorical_cols and numeric_cols:
                fig = px.pie(self.df, names=categorical_cols[0], values=numeric_cols[0], 
                             hole=0.3, title='Donut Chart Representation')
                return self._format_plot(fig)
            return None

        def data_table():
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(self.df.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[self.df[col] for col in self.df.columns],
                           fill_color='lavender',
                           align='left')
            )])
            fig.update_layout(title='Detailed Data Table')
            return self._format_plot(fig)

        # Visualization selection logic
        visualizations = [
            (line_chart, 'time series' in query.lower() or 'trend' in query.lower()),
            (bar_chart, 'compare' in query.lower() or 'comparison' in query.lower()),
            (pie_chart, 'proportion' in query.lower() or 'percentage' in query.lower()),
            (area_chart, 'cumulative' in query.lower() or 'total' in query.lower()),
            (donut_chart, 'breakdown' in query.lower() or 'parts' in query.lower()),
            (data_table, 'table' in query.lower() or 'details' in query.lower())
        ]

        # Try to find a matching visualization
        for (viz_func, condition) in visualizations:
            if condition:
                result = viz_func()
                if result:
                    return result

        # Fallback to default visualization
        return default_visualization()

    def _format_plot(self, fig):
        """
        Format the plot with enhanced styling and interactivity
        """
        fig.update_layout(
            plot_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12),
            title_font=dict(size=16, family='Arial, bold'),
            margin=dict(l=50, r=50, t=50, b=50),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial, sans-serif"
            )
        )
        
        # Convert plot to base64 for easy embedding
        buffer = io.BytesIO()
        fig.write_image(buffer, format="png", scale=2)
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return plot_base64

def extract_text_from_csv(csv_path: str) -> str:
    """Extracts and formats text from a CSV file with a fallback encoding."""
    text = ""
    try:
        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                text += " ".join(row) + "\n"
    except UnicodeDecodeError:
        # Retry with a different encoding if utf-8 fails
        with open(csv_path, mode='r', encoding='ISO-8859-1') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                text += " ".join(row) + "\n"
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
    You are an advanced information extraction and visualization assistant. Your tasks include:

    1. Comprehensively analyze the CSV data
    2. Answer user queries with detailed textual explanations
    3. Recommend and generate appropriate data visualizations
    4. Provide insights that go beyond simple data representation

    Visualization Recommendation Guidelines:
    - Identify the most suitable chart type based on the data:
      * Line Chart: For time series or trend data
      * Bar Chart: For comparing categories
      * Pie Chart: For showing proportions
      * Area Chart: For cumulative data
      * Donut Chart: For showing parts of a whole
      * Table: For detailed data representation

    - Consider data characteristics like:
      * Number of categories
      * Numeric vs categorical data
      * Temporal aspects
      * Relationships between variables

    Context:
    {context}

    <</SYS>>

    User Query: {question}

    Provide a comprehensive response that includes:
    1. Detailed textual explanation
    2. Recommended visualization type
    3. Specific insights from the data

    [/INST]"""
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def save_summary_and_interaction_to_mongodb(csv_summary, interactions, visualization=None):
    """Save the CSV summary, interactions, and visualization to MongoDB."""
    document = {
        "csv_summary": csv_summary,
        "interactions": interactions,
        "visualization": visualization
    }
    collection.insert_one(document)
    print("Summary, interactions, and visualization saved to MongoDB.")

async def main():
    csv_path = input("Enter the path to the CSV file: ").strip()

    try:
        # Read CSV using pandas for better data handling
        df = pd.read_csv(csv_path)
        
        # Create visualization generator
        viz_generator = VisualizationGenerator(df)
        
        # Extract text for summarization and vector indexing
        csv_text = extract_text_from_csv(csv_path)
        print("CSV Text Extracted Successfully.")
        
        # Summarize the CSV content
        csv_summary = await summarize_large_text(csv_text)
        print("Summary of CSV:\n", csv_summary)
        
        # Create content chunks for vector indexing
        content_chunks = chunk_text(csv_text, max_length=2000)
        
        # Create FAISS index
        try:
            faiss_index = create_faiss_index(content_chunks)
            print("FAISS index created successfully.")
        except Exception as e:
            print(f"Failed to create FAISS index: {e}")
            return

        # Set up retrieval QA chain
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

        # Interaction loop
        interactions = []
        while True:
            user_query = input("Ask a question (or type 'exit'): ").strip()
            if user_query.lower() == 'exit':
                print("Thank you for using the assistant!")
                break

            try:
                # Get textual response
                result = qa_chain.invoke({"query": user_query})
                response = result['result']
                
                # Generate visualization
                visualization = viz_generator.generate_visualization(user_query)
                
                # Print response
                print(f"Response: {response}")
                
                # Save interaction with optional visualization
                interactions.append({
                    "query": user_query, 
                    "response": response,
                    "visualization": visualization
                })
                
                # Optional: Prompt to view visualization (you can customize this)
                view_viz = input("Would you like to view the visualization? (yes/no): ").strip().lower()
                if view_viz == 'yes' and visualization:
                    print("Visualization saved. You can retrieve it from the saved interactions.")
                
            except Exception as e:
                print(f"Error during query: {e}")

        # Save final summary and interactions to MongoDB
        save_summary_and_interaction_to_mongodb(csv_summary, interactions)

    except Exception as e:
        print(f"Error processing CSV: {e}")

if __name__ == "__main__":
    asyncio.run(main())