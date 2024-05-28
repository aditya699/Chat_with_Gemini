'''
In case any teacher wants to use translation
'''
import os
import openai
from langchain_openai import AzureOpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
os.environ["AZURE_OPENAI_ENDPOINT"]=os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"]=os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)
text=input("Enter the language you want to covert text into : ")
text_input=input("Enter the text : ")

message = HumanMessage(
    content=f"Translate {text_input} sentence from English to {text}"
)
print(model.invoke([message]).content)