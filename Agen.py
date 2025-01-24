import os
import pandas as pd
import warnings
from pandasai import Agent
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM 


warnings.filterwarnings('ignore', category=FutureWarning)

os.environ["PANDASAI_API_KEY"] = "$2a$10$QX0Dem6qQgVVv9u75EVFmeIB56Ka23.YNPxzOoxXcujK81udvUEf2"
path = input("File Path :")

def load_file(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    return None

Data = load_file(path)


agent = Agent(Data)

prompt = "Analyze this data and provide insights"
response = agent.call_llm_with_prompt(prompt)

while True:
    query = input("Enter Query:")
    if 'exit' == query:
        print("!!")
        break
    print(agent.run(Data, query))


