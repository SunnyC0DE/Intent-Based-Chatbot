import json
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

nltk.tokenize._get_punkt_tokenizer = lambda lang='english': PunktSentenceTokenizer()
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle

with open("data/intents.json", 'rb') as file:
    data = json.load(file)

nltk.download('punkt')
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)

        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

x_train = list(training[:,0])
y_train = list(training[:, 1])

pickle.dump(x_train, open('x_train.pkl','wb'))
pickle.dump(y_train, open('y_train.pkl','wb'))


pickle.dump(words, open('word.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))