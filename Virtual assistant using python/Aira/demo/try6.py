import datetime
import speech_recognition as sr
import pyautogui
import webbrowser as wb
import wikipedia
import pywhatkit
import AppKit
import requests
import clipboard
import pyjokes
import string
import random
import time as tt
from newsapi import NewsApiClient


def speak(text, speed=200, voice="com.apple.speech.synthesis.voice.Karen"):
    speech = AppKit.NSSpeechSynthesizer.alloc().init()
    speech.setVoice_(voice)
    speech.setRate_(speed)
    speech.startSpeakingString_(text)
    while speech.isSpeaking():
        continue

def screenshot():
    name_img = tt.time()
    name_img = 'C:\\Users\\khush\\OneDrive\\Desktop\\aira_2\\Exalt\\screenshots\\{name_img}.png'
    img = pyautogui.screenshot(name_img)
    img.show()

def time():
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    speak("The current time is", speed=150)
    pyautogui.sleep(2)
    speak(current_time, speed=150)

def news():
    newsapi = NewsApiClient(api_key='89b3165f0699445b8d69d61c1c2ac729')  # Remember to replace 'YOUR_NEWS_API_KEY' with your actual API key
    speak("What topic would you like to know about?")
    topic = takeCommandMIC()
    if topic != "None":
        data = newsapi.get_top_headlines(q=topic, language='en', page_size=5)
        newsdata = data['articles']
        if newsdata:  # Checking if there are articles available for the given topic
            for x, article in enumerate(newsdata, 1):
                speak(f"News {x}: {article['title']}")
                speak(article['description'])
        else:
            speak(f"Sorry, I couldn't find any news on {topic}.")
    else:
        speak("No topic provided. Please try again.")
    speak("That's it for now. I will update you in some time.")



def date():
    year = int(datetime.datetime.now().year)
    month = int(datetime.datetime.now().month)
    date = int(datetime.datetime.now().day)
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    month_name = datetime.date(1900, month, 1).strftime('%B')
    speak("Today is {} {} {}, and the current time is {}".format(date, month_name, year, current_time))

def passwordgen():
    s1 = string.ascii_uppercase
    s2 = string.ascii_lowercase
    s3 = string.digits
    s4 = string.punctuation

    passlen = 8
    s = []
    s.extend(list(s1))
    s.extend(list(s2))
    s.extend(list(s3))
    s.extend(list(s4))

    random.shuffle(s)
    newpass = ("".join(s[0:passlen]))
    print(newpass)
    speak(newpass)


def wishme():
    voice = "com.apple.speech.synthesis.voice.Karen"
    speak("My name is Med-IQ advisor, How may i assist you", voice=voice)
    pyautogui.sleep(2)
    
    hour = datetime.datetime.now().hour
    if hour >= 6 and hour <12:
        speak("Good Morning ma'am!", voice=voice)
    elif hour >=12 and hour <18:
        speak("Good afternoon ma'am!", voice=voice)
    elif hour >=18 and hour <24:
        speak("Good Evening ma'am!", voice=voice)
    else:
       speak("Aira at your service, please tell me how can I help you?", voice=voice)
       


def takeCommandMIC():
    r= sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening....")
        r.pause_threshold = 1
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            speak("Sorry, I didn't hear anything. Please try again.")
            return "None"
        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language="en-IN")
            print("user said:" +query)
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand what you said. Please try again.")
            return "None"
        except sr.RequestError:
            speak("Sorry, there was an error in the speech recognition service. Please try again later.")
            return "None"
    return query.lower()



def sendwhatsmsg(phone_no, message):
    try:
        url = f"https://web.whatsapp.com/send?phone={phone_no}&text={message}"
        chrome_path = "open -a /Applications/Google\ Chrome.app %s"
        wb.get(chrome_path).open(url)
        pyautogui.sleep(10)
        pyautogui.press('enter')
        speak("Message has been sent.")
    except Exception as e:
        print(e)
        speak("Unable to send the message.")


def searchgoogle():
    speak('What should I search for?')
    search = takeCommandMIC()
    if search != "None":
        wb.open(f'https://www.google.com//search?q={search}')

def roll():
    speak("okay sir, rolling a die for you")
    die = ['1', '2', '3', '4','5','6']
    roll = []
    roll.extend(die)
    random.shuffle(roll)
    roll = ("".join(roll[0]))
    speak("I rolled a die and you got"+roll)

def anysearch():
    speak('You are on the way please say in detail again?')
    search = takeCommandMIC()
    if search != "None":
        wb.open(f'https://www.google.com//search?q={search}')


def text2speech():
    text = clipboard.paste()
    print(text)
    speak(text)

def flip():
    speak("okay sir, flipping a coin")
    coin = ('head', 'tails')
    toss = []
    toss.extend(coin)
    random.shuffle(toss)
    toss = ("".join(toss[0]))
    speak("I flipped the coin and you got" + toss)

if __name__ == "__main__":
    wishme()
    while True:
        query = takeCommandMIC().lower()

        if 'time' in query:
            time()

        elif 'date' in query:
            date()

        elif 'message' in query:
            user_name = {
               'computer':'+91 9665596851'
            }
            try:
               speak("to whom you want to send the message? ")
               name = takeCommandMIC()
               phone_no = user_name[name]
               speak("what is the message?")
               Message = takeCommandMIC()
               sendwhatsmsg(phone_no, Message)
               speak("message has been send")
            except Exception as e:
               print(e)
               speak("unable to send the Message")
               

        elif 'wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=2)
            speak("According to Wikipedia")
            print(results)
            speak(results)
            
            
        elif 'open google' in query:
            searchgoogle()


        elif 'open youtube' in query:
            speak("What should I search on youtube")
            topic = takeCommandMIC()
            pywhatkit.playonyt(topic)


        
        elif 'weather' in query:
            city = 'Thane'
            url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&units=imperial&appid=382962ca87c1d84b2159fe2425532d8f'
            
            res = requests.get(url)
            data = res.json()

            weather = data['weather'][0]['main']
            temp = data['main']['temp']
            desp = data['weather'][0]['description']
            temp = round((temp - 32) *5/9)
            print(weather)
            print(temp)
            print(desp)
            speak(f'weather in {city} is like')
            speak('Temperature : {} degree celcius'. format(temp))
            speak('weather is {}'.format(desp))


        elif 'joke' in query:
            speak(pyjokes.get_joke())

        elif 'read' in query:
            text2speech()

        elif 'password' in query:
            passwordgen()

        elif 'news' in query:
            news()

        elif 'screenshot' in query:
            screenshot()
        
        elif 'please remember' in query:
            speak("what should i remember?")
            data = takeCommandMIC()
            speak("you told me to remember that"+data)
            remember = open('data.txt','w')
            remember.write(data)
            remember.close()

        elif 'flip' in query:
            flip()
        
        elif 'do you remember anything' in query:
            remember = open('data.txt', 'r')
            speak("you told me to remember that"+remember.read())
        elif 'name' in query:
            speak("My name is MED-IQ advisor. How may I assist you?")
        
        elif 'for me' in query:
            speak("I can perform more than 50 task, which will defenetely help you in your day to day life ")

        elif 'age' in query:
            speak("my age is 19 and i am studying in python school")

        elif 'stay' in query:
            speak("I stay in thane")

        elif 'roll' in query:
            roll()

        


#question based on what, where, how, when......




        elif 'what' in query:
            anysearch()

        elif 'can' in query:
            anysearch()

        elif 'where' in query:
            anysearch()

        elif 'when' in query:
            anysearch()

        elif 'how' in query:
            anysearch()

        elif 'which' in query:
            anysearch()

        elif 'why' in query:
            anysearch()

        elif 'who' in query:
            anysearch()

        elif 'is' in query:
            anysearch()

        elif 'the' in query:
            anysearch()

        elif 'say' in query:
            anysearch()

        elif 'in' in query:
            anysearch()

        elif 'details' in query:
            anysearch()

        elif 'do' in query:
            anysearch()

        elif 'you' in query:
            anysearch()

        elif 'ok' in query:
            anysearch()

        elif 'offline' in query:
            quit()