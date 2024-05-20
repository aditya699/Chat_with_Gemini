'''
This file will convert all the helper modules.

Modules-
1.Getting input by user using microphone (Aditya 7:38AM 25-03-2024)

'''
import speech_recognition as sr
import pyttsx3

# Initialize the recognizer
r = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set the speaking rate (default is around 200 words per minute)
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 90)  # Decrease the rate to make it speak more slowly

def get_text():
    while True:
        with sr.Microphone() as source:
            engine.say("Hi, what would you like to study or ask me today?")
            engine.runAndWait()
            
            print("Listening...")
            audio_text = r.listen(source)

            try:
                # Recognize the speech
                text = r.recognize_google(audio_text)
                print("I hope you said: " + text + ". If yes, say 'yes', else say 'no'.")
                
                # Respond
                engine.say("I hope you said: " + text)
                engine.say("If yes, say 'yes', else say 'no'")
                engine.runAndWait()

                # Listen for confirmation
                print("Listening for confirmation...")
                audio_confirm = r.listen(source)
                confirmation = r.recognize_google(audio_confirm).lower()


                if confirmation == "yes":
                    engine.say("Great! Let's proceed.")
                    engine.runAndWait()
                    return text
                else:
                    engine.say("Let's try again.")
                    engine.runAndWait()

            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
                engine.say("Sorry, I could not understand the audio.")
                engine.runAndWait()
            
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
                engine.say("Could not request results; {0}".format(e))
                engine.runAndWait()