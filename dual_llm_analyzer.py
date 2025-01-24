import os
import sys
import pandas as pd
import requests
from pandasai import SmartDataframe

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

API_BASE = "https://api.sambanova.ai/v1"
MODEL_405B = "Meta-Llama-3.1-405B-Instruct"
MODEL_70B = "Meta-Llama-3.1-70B-Instruct"
API_KEY = "540f8914-997e-46c6-829a-ff76f5d4d265"

def load_file(file_path):
    """Load CSV or Excel file and return as pandas DataFrame"""
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

def analyze_with_405b(df):
    """
    Use 405B model to identify data issues and generate analysis prompt
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Analyze this dataset and identify ALL possible data quality issues, inconsistencies, and potential problems. 
    For each issue found, you MUST specify the exact cell location (row and column) where the issue occurs.
    Be extremely thorough and detailed in your analysis.

Dataset :
{df.to_string()}

Dataset Info:
{df.info()}

Dataset Description:
{df.describe().to_string()}

Please analyze and return issues in this format:
[Row, Column] - Issue Description


Analyze for:
1. Missing Data:
   - List exact cells with missing values
   - Include row numbers and column names

2. Data Type Issues:
   - Identify specific cells with incorrect/mixed data types
   - Point out exact locations of encoding issues

3. Data Quality:
   - List duplicate record locations
   - Specify cells with inconsistent formatting
   - Identify exact locations of data entry errors

4. Statistical Anomalies:
   - Point out specific outlier cells
   - List cells involved in suspicious patterns
   - Identify exact locations of unusual values

5. Business Logic Issues:
   - Specify cells with logical inconsistencies
   - List exact locations of out-of-range values
   - Identify cells involved in impossible combinations

Group your findings by issue type and always include the exact [Row, Column] location for each issue."""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a data quality expert focused on thorough analysis and issue detection."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": MODEL_405B,
        "max_tokens": 4000,
        "temperature": 0.2
    }

    try:
        response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            print(f"405B API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error in 405B analysis: {str(e)}")
        return None

def highlight_issues_with_70b(df, issues_prompt):
    """
    Use 70B model to highlight and explain the issues in the original file
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Given this dataset and the identified issues, provide a detailed report highlighting and explaining each problem in the context of the data.



Identified Issues:
{issues_prompt}

For each issue:
1. Highlight the specific data points or columns affected
2. Explain the potential impact on data analysis
3. Provide specific examples from the dataset
4. Suggest potential solutions or corrections

Format your response as:
ISSUE LOCATION | DESCRIPTION | IMPACT | EXAMPLE | SOLUTION

Use markdown formatting to highlight critical issues."""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a data quality expert who explains and contextualizes data issues clearly."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": MODEL_70B,
        "max_tokens": 4000,
        "temperature": 0.3
    }

    try:
        response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            print(f"70B API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error in 70B analysis: {str(e)}")
        return None

def main():
    file_path = input("Enter the path to your data file (CSV or Excel): ").strip()
    
    df = load_file(file_path)
    if df is None:
        print("Failed to load file. Exiting.")
        return
    print("\nAnalyzing data with 405B model...")
    issues_prompt = analyze_with_405b(df)
    
    if issues_prompt is None:
        print("Failed to analyze data with 405B model. Exiting.")
        return
    
    print("\nIdentified Issues:")
    print(issues_prompt)
    
if __name__ == "__main__":
    main()
