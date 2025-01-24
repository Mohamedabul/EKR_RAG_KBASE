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

def detect_data_issues(df):
    """
    Analyze the DataFrame for common data issues and return a description of found issues.
    """
    issues = []
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        missing_cols = missing_values[missing_values > 0]
        issues.append(f"Missing values found in columns: {', '.join(missing_cols.index)} "
                     f"(counts: {', '.join(map(str, missing_cols.values))})")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate rows")
    
    # Check for inconsistent data types
    mixed_type_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column contains mixed numeric and non-numeric values
            numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
            if 0 < numeric_count < len(df):
                mixed_type_cols.append(col)
    if mixed_type_cols:
        issues.append(f"Mixed data types found in columns: {', '.join(mixed_type_cols)}")
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)].shape[0]
        if outliers > 0:
            issues.append(f"Found {outliers} potential outliers in column '{col}'")
    
    # Check for inconsistent string formatting
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        # Check for mixed case
        if df[col].str.isupper().any() and df[col].str.islower().any():
            issues.append(f"Inconsistent string casing found in column '{col}'")
        
        # Check for leading/trailing spaces
        if (df[col].str.len() != df[col].str.strip().str.len()).any():
            issues.append(f"Leading or trailing spaces found in column '{col}'")
    
    if not issues:
        return "No significant data issues found."
    
    return "Data Issues Found:\n" + "\n".join(f"- {issue}" for issue in issues)

def detect_issues_with_llama(data):
    """
    Use LLaMA model to detect issues in the provided data
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Analyze the following dataset and identify ALL potential data quality issues, inconsistencies, and areas for improvement. 
    Be thorough and specific in your analysis.

    Dataset Preview:
    {data.to_string()}

    Dataset Info:
    {data.info()}

    Please identify:
    1. Data completeness issues (missing values, incomplete records)
    2. Data consistency issues (format inconsistencies, contradictions)
    3. Data accuracy issues (outliers, incorrect values)
    4. Data format issues (incorrect data types, formatting problems)
    5. Any other relevant quality concerns

    Return ONLY the list of issues in a clear, structured format."""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a data quality expert. Analyze the provided dataset and return a detailed list of all data quality issues."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": MODEL,
        "max_tokens": 2000,
        "temperature": 0.3
    }

    try:
        response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            return "Could not detect issues using LLaMA model"
        
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error in LLaMA issue detection: {str(e)}")
        return "Error occurred while detecting issues"

def process_with_llama(original_data, query, query_result):
    """
    Process data with LLaMA 70B model and return modified data
    """
    # Detect data issues using LLaMA
    data_issues = detect_issues_with_llama(original_data)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Based on the following data, identified issues, and query, provide a MODIFIED version of the original data that addresses both the identified issues and the query. Return ONLY the modified data in a table format.

Original Data:
{original_data.to_string()}

LLaMA-Identified Data Issues:
{data_issues}

User Query:
{query}

Initial Query Result:
{query_result.to_string() if isinstance(query_result, pd.DataFrame) else str(query_result)}

IMPORTANT INSTRUCTIONS:
1. Address ALL the identified issues in your response
2. Return the complete modified dataset as a table
3. Include ALL relevant columns from the original data
4. Make necessary modifications based on both the issues and query
5. Use the same column names as the original data
6. Maintain proper data types for each column
7. Use '|' as column separator
8. Include a header row

Return ONLY the modified table with no additional text."""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a data transformation assistant. Return only the modified data table with the same structure as the input data."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": MODEL,
        "max_tokens": 4000,
        "temperature": 0.3  
    }

    try:
        response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        result = response.json()['choices'][0]['message']['content']
        table_lines = [line.strip() for line in result.strip().split('\n') if '|' in line]
        if not table_lines:
            return None
            
        # Convert the table string back to DataFrame
        try:
            # Split the header and clean it
            headers = [col.strip() for col in table_lines[0].split('|') if col.strip()]
            
            # Process data rows
            data = []
            for line in table_lines[2:]:  # Skip header and separator line
                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                if row:  # Only add non-empty rows
                    data.append(row)
            
            # Create DataFrame with the same structure as original
            modified_df = pd.DataFrame(data, columns=headers)
            
            # Convert data types to match original DataFrame
            for col in modified_df.columns:
                if col in original_data.columns:
                    try:
                        modified_df[col] = modified_df[col].astype(original_data[col].dtype)
                    except:
                        pass  # Keep original type if conversion fails
            
            return modified_df
            
        except Exception as e:
            print(f"Error converting table to DataFrame: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error in LLaMA processing: {str(e)}")
        return None

def main():
    llm = LocalLLM(api_base=API_BASE, model=MODEL, api_key=API_KEY)
    
    print("Please enter file paths one at a time. Type 'done' when finished.")
    dataframes = load_multiple_files()
    
    if not dataframes:
        print("No valid files were loaded. Exiting.")
        return

    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    
    # Display initial data analysis
    # print("\n" + "="*50)
    # print("INITIAL DATA ANALYSIS")
    # print("="*50)
    # print("\nOriginal Data Preview:")
    # print(combined_df.head().to_string())
    print("\nData Shape:", combined_df.shape)
    
    # Detect and display issues
    issues = detect_data_issues(combined_df)
    print("\n" + "="*50)
    print("DETECTED DATA ISSUES")
    print("="*50)
    print(issues)
    
    custom_prompt = """You are a helpful assistant capable of processing and returning data in tabular form. Whenever a user asks for information, process the query and return the result as a well-structured table. Please ensure that the table includes the relevant columns, data, and is easy to read.

            Structure: Format the result as a table with clear headers.
            Data Representation: Ensure the data is returned in a structured, tabular form, with appropriate rows and columns.
            Context Understanding: If the query is about filtering, sorting, or aggregating data, return the modified table as per the user's request.
            Handle Missing Data: If applicable, indicate missing data as NaN or similar.
            Respond with the Table Only: Focus on returning the table, without unnecessary explanation.
    Query: {question}
    Please provide a clear and relevant response."""
    
    smart_df = SmartDatalake(
        combined_df, 
        config={
            "llm": llm,
            "prompt_template": custom_prompt
        }
    )

    while True:
        print("\n" + "="*50)
        print("QUERY PROCESSING")
        print("="*50)
        query = input("Enter Query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        else:
            try:
                print("\nProcessing query...")
                query_result = smart_df.chat(query)
                print("\nInitial Query Result:")
                print("-"*30)
                print(query_result)
                
                print("\nProcessing with LLaMA for improvements...")
                modified_df = process_with_llama(combined_df, query, query_result)
                
                if modified_df is not None:
                    print("\n" + "="*50)
                    print("FINAL MODIFIED DATA")
                    print("="*50)
                    print("\nChanges made to address both data issues and query:")
                    print(modified_df.to_string())
                    
                    # Show what changed
                    print("\nSummary of Changes:")
                    print("-"*30)
                    
                    # Compare shapes
                    if modified_df.shape != combined_df.shape:
                        print(f"- Row count changed from {combined_df.shape[0]} to {modified_df.shape[0]}")
                    
                    # Compare missing values
                    original_missing = combined_df.isnull().sum().sum()
                    modified_missing = modified_df.isnull().sum().sum()
                    if original_missing != modified_missing:
                        print(f"- Missing values changed from {original_missing} to {modified_missing}")
                    
                    # Compare duplicates
                    original_duplicates = combined_df.duplicated().sum()
                    modified_duplicates = modified_df.duplicated().sum()
                    if original_duplicates != modified_duplicates:
                        print(f"- Duplicate rows changed from {original_duplicates} to {modified_duplicates}")
                    
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