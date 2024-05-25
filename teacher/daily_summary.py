'''
Author -Aditya Bhatt 25-05-2024 10:27AM

Obejctive-
Create a reponse summary for teachers's assisment of what students are actually asking
'''
import os
import getpass
from dotenv import load_dotenv
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import speech_recognition as sr
import pyttsx3
from langchain_openai import ChatOpenAI
from datetime import datetime
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys from the environment variables
google_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['GOOGLE_API_KEY'] = google_api_key 
os.environ['OPENAI_API_KEY'] = openai_api_key

# Get today's date
today = datetime.today().date()

# Read the CSV file containing the questions and responses
data_dir = './data/'
csv_file = os.path.join(data_dir, f'combined_log_{today}.csv')

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file)



# Initialize a list to hold the summaries
questions=[]
answers=[]
# Process each question and response
for index, row in df.iterrows():
    questions.append(row['Student Question'])
    answers.append(row['Response'])

summary=questions+answers
summary_context=' '.join(summary)
# Define the prompt template and output parser
prompt_one = ChatPromptTemplate.from_template("Here is the conversation of students and chatbots {summary} summarize in a concise and clear manner, ensuring the teacher can know more about her students queries and chatbots responses in a paragragp manner")
output_parser = StrOutputParser()

# Initialize language models
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, safety_settings=None)
llm_1 = ChatOpenAI(model="gpt-3.5-turbo")

# Define the processing chains
chain = prompt_one | llm    | output_parser
# Define the processing chains
chain1 = prompt_one | llm_1    | output_parser

response =chain.invoke(f"{summary_context}")

if len(response) ==0:
    response=chain1.invoke(f"{summary_context}")

print(response)

# Create file name dynamically
file_name = f"data/conversation_summary_{today}.txt"

# Write the response to a text file
with open(file_name, 'w') as file:
    file.write(response)

print(f"Conversation summary has been saved to '{file_name}'.")