import os
import sys
import pandas as pd
import requests
from pandasai import SmartDataframe, SmartDatalake
from pandasai.llm.local_llm import LocalLLM 

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

API_BASE = "https://api.sambanova.ai/v1"
MODEL = "Meta-Llama-3.1-70B-Instruct"
API_KEY = "540f8914-997e-46c6-829a-ff76f5d4d265"

def load_file(file_path):
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        else:
            print("Unsupported file format. Please use CSV or Excel files.")
            return None
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None

def load_multiple_files():
    dataframes = []
    while True:
        file_path = input("Enter File Path (or 'done' when finished): ").strip()
        if file_path.lower() == 'done':
            break
        
        df = load_file(file_path)
        if df is not None:
            dataframes.append(df)
            print(f"Successfully loaded: {file_path}")
    
    return dataframes


def main():
    print("Please enter file paths one at a time. Type 'done' when finished.")
    dataframes = load_multiple_files()
    
    if not dataframes:
        print("No valid files were loaded. Exiting.")
        return

    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    
    # Detect data issues
    # print("\nAnalyzing data issues...")
    # data_issues = detect_data_issues(combined_df)
    # print("\nData Analysis Results:")
    # print(data_issues)
    
    # # Process with LLaMA to get modified data
    # print("\nProcessing data with LLaMA to address identified issues...")
    # modified_df = process_with_llama(combined_df, data_issues)
    
    # if modified_df is not None:
    #     print("\nModified Data from LLaMA:")
    #     print(modified_df.to_string())
    #     print("\nData has been processed and issues have been addressed.")
    # else:
    #     print("\nCould not get modified data from LLaMA.")

    custom_prompt = """You are a helpful assistant capable of processing and returning data in tabular form. Whenever a user asks for information, process the query and return the result as a well-structured table. Please ensure that the table includes the relevant columns, data, and is easy to read.

            Structure: Format the result as a table with clear headers.
            Data Representation: Ensure the data is returned in a structured, tabular form, with appropriate rows and columns.
            Context Understanding: If the query is about filtering, sorting, or aggregating data, return the modified table as per the user's request.
            Handle Missing Data: If applicable, indicate missing data as NaN or similar.
            Respond with the Table Only: Focus on returning the table, without unnecessary explanation.
    Query: {question}
    Please provide a clear and relevant response."""
    
    llm = LocalLLM(api_base=API_BASE, model=MODEL, api_key=API_KEY)
    smart_df = SmartDatalake(
        combined_df, 
        config={
            "llm": llm,
            "prompt_template": custom_prompt
        }
    )

    while True:
        query = input("Enter Query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        else:
            try:
                query_result = smart_df.chat(query)
                print("\nInitial Query Result:", query_result)
                
                modified_df = process_with_llama(combined_df, data_issues)
                
                if modified_df is not None:
                    print("\nModified Data from LLaMA:")
                    print(modified_df.to_string())
                    # Update the combined_df with modified data
                    combined_df = modified_df
                    print("\nData has been updated with the modifications.")
                else:
                    print("\nCould not get modified data from LLaMA.")
                    
            except Exception as e:
                error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
                print(f"Error: {error_msg}")

if __name__ == "__main__":
    main()