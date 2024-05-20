import os
import getpass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src import utils

#Load environment variables from .env file
load_dotenv()

#Retrieve the API key from the environment variable
google_api_key = os.getenv("GEMINI_API_KEY")
os.environ['GOOGLE_API_KEY']=google_api_key 
llm = ChatGoogleGenerativeAI(model="gemini-pro")

query=utils.get_text()
print(query)


# os.environ['GOOGLE_API_KEY']=google_api_key 
# llm = ChatGoogleGenerativeAI(model="gemini-pro")
# result = llm.invoke("Write a ballad about LangChain")
# print(result.content)