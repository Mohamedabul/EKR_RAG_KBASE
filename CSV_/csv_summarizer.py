import pandas as pd  
import requests
import os

API_KEY = os.getenv("SAMBA_API_KEY", "540f8914-997e-46c6-829a-ff76f5d4d265")
API_URL = os.getenv("SAMBA_API_URL", "https://api.sambanova.ai/v1/chat/completions")

def extract_text_from_csv(csv_path: str) -> str:
    """Extracts key information from the CSV."""
    try:
        df = pd.read_csv(csv_path, encoding='latin-1')
        
        # Create a brief description of the dataset
        description = (
            f"Dataset Summary:\n"
            f"- Total Rows: {len(df)}\n"
            f"- Total Columns: {len(df.columns)}\n"
            f"- Column Names: {', '.join(df.columns)}\n"
        )
        return description

    except Exception as e:
        raise Exception(f"Failed to extract data from CSV: {e}")

def summarize_text(text: str) -> str:
    """Uses SambaNova API to summarize the given text about the dataset."""
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

    response = requests.post(API_URL, json=payload, headers=headers, timeout=90)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    csv_path = input("Enter the path to the CSV file: ").strip()

    try:
        extracted_text = extract_text_from_csv(csv_path)
        print("CSV Data Extracted Successfully.")
        print("Extracted Content:\n", extracted_text)

        # Summarize the dataset using the SambaNova API
        summary = summarize_text(extracted_text)
        print("\nSummary:\n", summary)

    except Exception as e:
        print(f"An error occurred: {e}")
