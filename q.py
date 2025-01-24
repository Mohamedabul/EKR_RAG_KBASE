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
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    json_format = '''{
  "defects": [
    {
      "column": "Column Name",
      "issues": [
        {
          "value": "TBD/Blank/Inconsistent",
          "rows": ["Affected Application Names"]
        }
      ]
    }
  ]
}'''
    
    prompt = f"""Analyze the following CSV data for data quality issues. 
Categorize defects by column, identifying:
- Missing or TBD values
- Inconsistent formatting
- Blank entries
- Unusual patterns
 
Provide output in this structured JSON format:
{json_format}
 
CSV Data: {df}"""

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
        sys.exit(1)
        
    issues_prompt = analyze_with_405b(df)
    
    if issues_prompt is None:
        sys.exit(1)
    
    print(issues_prompt)  

main()