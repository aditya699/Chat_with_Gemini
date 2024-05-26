'''
Author - Aditya Bhatt 26-05-2024 8:22 PM

Objective -
1.Create a system for students to chat with ncert based data.

For Demo Purpose we have taken Study Civics book for Class 8
'''
import os
import getpass
from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# Retrieve the API key from the environment variable
google_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['GOOGLE_API_KEY'] = google_api_key 
os.environ['OPENAI_API_KEY'] = openai_api_key

from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-pro")

loader = PyPDFLoader("C:/Users/aditya/Desktop/2024/Inclusive.AI/Study Material/civics.pdf")
pages = loader.load_and_split()
docs=[pages[0]]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)



