'''
In case any teacher wants to use translation
'''



import os
import openai
from langchain_openai import AzureOpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-06-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"]="https://aiagents.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"]="198e8394cb0f47e69807d2bcd7a26c60"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "test_4o"
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