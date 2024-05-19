import os
import getpass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import speech_recognition as sr
import pyttsx3

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()
# r.energy_threshold=10
# Initialize text-to-speech engine
engine = pyttsx3.init()

with sr.Microphone() as source:
        
        engine.say("Hi what would you like to study today?")
        engine.runAndWait()
        audio_text = r.listen(source)

        print(audio_text)

        engine.say(f"{audio_text}")

        engine.say("thanks")
        engine.runAndWait()
# # Load environment variables from .env file
# load_dotenv()
# # Retrieve the API key from the environment variable
# google_api_key = os.getenv("GEMINI_API_KEY")

# os.environ['GOOGLE_API_KEY']=google_api_key 
# llm = ChatGoogleGenerativeAI(model="gemini-pro")
# result = llm.invoke("Write a ballad about LangChain")
# print(result.content)