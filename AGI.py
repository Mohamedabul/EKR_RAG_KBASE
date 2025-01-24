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
        if df[col].str.isupper().any() and df[col].str.islower().any():
            issues.append(f"Inconsistent string casing found in column '{col}'")
        if (df[col].str.len() != df[col].str.strip().str.len()).any():
            issues.append(f"Leading or trailing spaces found in column '{col}'")
    
    return "Data Issues Found:\n" + "\n".join(f"- {issue}" for issue in issues) if issues else "No significant data issues found."

def process_with_llama(original_data, data_issues):
    """
    Process data with LLaMA 70B model and return modified data based on identified issues
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Analyze the following dataset and provide a COMPREHENSIVE analysis of all data quality issues, followed by the modified data that addresses these issues. 

Original Data:
{original_data.to_string()}

Automated Analysis Results:
{data_issues}

Dataset Info:
{original_data.info()}

Please provide your analysis in the following format:

1. DETAILED DATA QUALITY ANALYSIS:
   a) Missing Values Analysis:
      - Identify columns with missing values
      - For numeric columns: Show median values that should be used for imputation
      - For categorical columns: Suggest appropriate handling strategies
   
   b) Data Type Issues:
      - Identify columns with inconsistent data types
      - Recommend proper data type conversions
   
   c) Outlier Analysis:
      - List all outliers found in numeric columns
      - Provide statistical context (e.g., how many standard deviations from mean)
      - Suggest whether outliers are likely errors or valid data points
   
   d) Format Inconsistencies:
      - Detail any inconsistent string formatting
      - Highlight inconsistent date formats, if any
      - Point out any standardization needs
   
   e) Data Validation Issues:
      - Identify any values outside valid ranges
      - Flag logically inconsistent data
      - Note any business rule violations

2. MODIFIED DATA TABLE:
After the analysis, provide the complete modified dataset with all issues addressed. Format as a table with:
- Use '|' as column separator
- Include header row
- Maintain original column names
- Apply all suggested fixes

Return the analysis followed by the modified table, separated by three dashes (---).
"""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a data quality expert. Provide detailed analysis of data issues and return the corrected data."
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
        
        # Split the response into analysis and table parts
        parts = result.split('---')
        if len(parts) >= 2:
            print("\nDetailed Analysis from LLaMA:")
            print(parts[0].strip())
            
            # Process the table part
            table_lines = [line.strip() for line in parts[1].strip().split('\n') if '|' in line]
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
                
        else:
            print("Invalid response format from LLaMA")
            return None
            
    except Exception as e:
        print(f"Error in LLaMA processing: {str(e)}")
        return None

def main():
    print("Please enter file paths one at a time. Type 'done' when finished.")
    dataframes = load_multiple_files()
    
    if not dataframes:
        print("No valid files were loaded. Exiting.")
        return

    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    
    # Detect data issues
    print("\nAnalyzing data issues...")
    data_issues = detect_data_issues(combined_df)
    print("\nData Analysis Results:")
    print(data_issues)
    
    # Process with LLaMA to get modified data
    print("\nProcessing data with LLaMA to address identified issues...")
    modified_df = process_with_llama(combined_df, data_issues)
    
    if modified_df is not None:
        print("\nModified Data from LLaMA:")
        print(modified_df.to_string())
        print("\nData has been processed and issues have been addressed.")
    else:
        print("\nCould not get modified data from LLaMA.")

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