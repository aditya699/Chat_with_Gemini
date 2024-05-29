'''
Author - Aditya Bhatt 26-05-2024 8:22 PM

Objective -
1.Create a system for students to chat with ncert based data.

For Demo Purpose we have taken Civics book for Class 8(NCERT)
'''
import PyPDF2
import os
import getpass
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import chainlit as cl
from datetime import datetime

# Retrieve the API key from the environment variable
google_api_key = os.getenv("GEMINI_API_KEY")
os.environ['GOOGLE_API_KEY'] = google_api_key 

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PdfReader(file)
        
        # Get the number of pages in the PDF
        num_pages = len(pdf_reader.pages)
        
        # Extract text from each page
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        
        return text
    
# Example usage
pdf_path = 'books/civics.pdf'
text = extract_text_from_pdf(pdf_path)
# print(text)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
chunks = text_splitter.split_text(text)

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
vector_store = FAISS.from_texts(chunks, embedding=embeddings)
vector_store.save_local("faiss_index")

retriever = vector_store.as_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, safety_settings=None)

template = """Act as friendly chatbot and give answer based on your knowledge and use the below context also.

{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


@cl.on_message
async def main(user_message:cl.Message):

    user_input = user_message.content
    response = rag_chain.invoke(user_input)
    
    # Send a response back to the user
    await cl.Message(
        content=f"Received: {response}",
    ).send()