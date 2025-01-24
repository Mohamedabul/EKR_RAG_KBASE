import pandas as pd  
import requests
import os

API_KEY = os.getenv("SAMBA_API_KEY", "540f8914-997e-46c6-829a-ff76f5d4d265")
API_URL = os.getenv("SAMBA_API_URL", "https://api.sambanova.ai/v1/chat/completions")

def extract_text_from_excel(excel_path: str) -> str:
    """Extracts key data from all sheets in an Excel file."""
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
    """Uses SambaNova API to summarize the given text."""
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

if __name__ == "__main__":
    file_path = input("Enter the path to the Excel file (.xls or .xlsx): ").strip()
    file_type = file_path.split(".")[-1].lower()

    try:
        if file_type not in ["xls", "xlsx"]:
            raise ValueError("Unsupported file type! Please provide an Excel file (.xls or .xlsx).")

        extracted_text = extract_text_from_excel(file_path)
        print("Excel Data Extracted Successfully.")
        print("Extracted Content:\n", extracted_text)

        summary = summarize_text(extracted_text)
        print("\nSummary:\n", summary)

    except Exception as e:
        print(f"An error occurred: {e}")
