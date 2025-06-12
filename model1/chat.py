import random
import json
import pickle
import numpy as np
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import requests
import webbrowser


from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
nltk.tokenize._get_punkt_tokenizer = lambda lang='english': PunktSentenceTokenizer()

nltk.download('punkt')
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

intents = json.load(open('data/intents.json'))
words = pickle.load(open('word.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')

def get_current_time():
    return "Current time is: " + datetime.now().strftime("%H:%M:%S")

def get_weather(city):
    url = f"https://wttr.in/{city}?format=j1" 
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        try:
            weather_data = {
                'city': city,
                'temperature': data['current_condition'][0]['temp_C'],
                'description': data['current_condition'][0]['weatherDesc'][0]['value'],
                'icon': data['current_condition'][0]['weatherIconUrl'][0]['value']
            }
            return f"The weather in {city} is {weather_data['description']} with {weather_data['temperature']}°C."
        except (KeyError, IndexError):
            return "Sorry, something went wrong while parsing the weather data."
    else:
        return "Sorry, couldn't fetch the weather."
    
def perform_google_search(query):
    webbrowser.open(f"https://www.google.com/search?q={query}")
    return "Here's what I found for you online!"

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    
    return np.array(bag)

def predict_class(sentence):
    bow_vector = bow(sentence, words)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x:x[1], reverse = True)
    return_list = []

    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

def get_response(ints, intents_json, user_message):
    tag = ints[0]['intent']
    for intent in intents_json['intents']:
        if tag == "get_time":
            return get_current_time()
        elif tag == "get_weather":
            return get_weather(city="Patiala")
        elif tag == "web_search":
            return perform_google_search(user_message)
        elif intent['tag'] == tag:
            return random.choice(intent['responses'])
        
def chat():
    print("Start Talking with the AI (Type 'quite' to stop)")

    while True:
        message = input("YOU: ")
        if message.lower() == "quite":
            break
        ints = predict_class(message)
        if ints:
            res = get_response(ints, intents, message)

        else:
            res = "sorry I did not understand"

        print("Bot: ",res)

chat()