'''
Features -
1.Basic ask and response via gemini.This will help students get response in a efficent manner.
'''



import os
import getpass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src import utils
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import speech_recognition as sr
import pyttsx3

# Initialize the recognizer
r = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

#Load environment variables from .env file
load_dotenv()

#Retrieve the API key from the environment variable
google_api_key = os.getenv("GEMINI_API_KEY")
os.environ['GOOGLE_API_KEY']=google_api_key 
llm = ChatGoogleGenerativeAI(model="gemini-pro")

start_of_conversation=utils.get_text()
prompt_one = ChatPromptTemplate.from_template("Explain the following {concept} to a blind student in a concise and clear manner, ensuring that the output is simple summary paragraph which doesn't contain any special chracters")
output_parser = StrOutputParser()
chain = prompt_one | llm | output_parser
result=chain.invoke(f"{start_of_conversation}")
engine.say(result)
engine.runAndWait()
