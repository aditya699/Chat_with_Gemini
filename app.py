import os
import getpass
from dotenv import load_dotenv
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from src import utils
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import speech_recognition as sr
import pyttsx3
from langchain_openai import ChatOpenAI
import datetime
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
# Initialize the recognizer
r = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
google_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['GOOGLE_API_KEY'] = google_api_key 
os.environ['OPENAI_API_KEY'] = openai_api_key

# Define the prompt template and output parser
prompt_one = ChatPromptTemplate.from_template("Explain the following {question} in a concise and clear manner, ensuring that the output is simple summary paragraph which has no special characters")
output_parser = StrOutputParser()

# Initialize language models
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, safety_settings=None)
llm_1 = ChatOpenAI(model="gpt-3.5-turbo")

# Define the processing chains
chain = prompt_one | llm | output_parser
chain2 = prompt_one | llm_1 | output_parser

# Create a DataFrame to store conversation logs
columns = ['Timestamp', 'Student Question', 'Response', 'Source']
log_df = pd.DataFrame(columns=columns)

while True:
    start_of_conversation = utils.get_text()
    if start_of_conversation == 'quit':
        break
    else:
        timestamp = datetime.datetime.now()
        result = chain.invoke(f"{start_of_conversation}")
        source = "Google Gemini"

        if len(result) == 0:
            result = chain2.invoke(f"{start_of_conversation}")
            source = "OpenAI GPT-3.5"

        # Log the conversation
        log_entry = pd.DataFrame([{
            'Timestamp': timestamp,
            'Student Question': start_of_conversation,
            'Response': result,
            'Source': source
        }])
        log_df = pd.concat([log_df, log_entry], ignore_index=True)

        # Output the response via TTS
        engine.say(result)
        engine.runAndWait()

# Save the log DataFrame to a CSV file with timestamp in the filename
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'conversation_log_{timestamp_str}.csv'
log_df.to_csv(filename, index=False)
