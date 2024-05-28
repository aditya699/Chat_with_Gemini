'''
Objective -
1.Create audio summary for all chapters
'''
import PyPDF2
import os
import getpass
from dotenv import load_dotenv
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
from gtts import gTTS
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

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

template = """Use the following pieces of context to summarize the text, and add relevant text based on your understanding of the question.
Always return a paragraphs with no special chracters

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

list_of_chapters=['On Equality','Role of the Government in Health','Growing up as Boys and Girls']
for i in list_of_chapters:
    
    print(f"Summary on chapter {i}")
    mytext=rag_chain.invoke(i)
    # Language in which you want to convert
    language = 'en'

    # Passing the text and language to the engine, 
    # here we have marked slow=False. Which tells 
    # the module that the converted audio should 
    # have a high speed
    myobj = gTTS(text=mytext, lang=language, slow=False)

    # Saving the converted audio in a mp3 file named
    # welcome 
    myobj.save(f"audio_summary/summary_{i}_english.mp3")





