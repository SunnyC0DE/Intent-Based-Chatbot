# 🤖 TensorFlow-Based Intent Chatbot (NLP + Classification)

An intelligent chatbot built using TensorFlow and NLP techniques for classifying intents and responding accordingly. It can answer greetings, provide time, weather updates, and perform Google search — all powered by a trained neural network model.

---

## 📌 Description

This chatbot uses natural language preprocessing (lemmatization, tokenization, lowercasing) to classify user input into predefined intents using a **sequential neural network model (ANN)** built in TensorFlow/Keras.

It then responds accordingly with a message or an action (like telling the current time, searching Google, or checking weather).

---

## 🧠 Features

- Trained intent classifier using TensorFlow
- JSON-based intent structure
- NLP preprocessing using NLTK
- Handles:
  - Greetings and farewells
  - Current time and date
  - Weather via API
  - Google search using `webbrowser`
- Extendable by adding more intents and retraining the model

---

## 📂 Project Structure


Model1
|_             data
|_             __pycache__
|_             chat.py
|_           chatbot_model.h5
|_            classes.pkl
|_           model.py
|_           preprocessing.py
|_            word.pkl
|_           x_train.pkl
|_           y_train.pkl



data
|_           intents.json



---

## 🔧 Technologies Used

- Python 3
- TensorFlow / Keras
- NLTK (`punkt`, `wordnet`, `lemmatizer`)
- NumPy
- Pickle
- Webbrowser / datetime / requests

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/SunnyC0DE/Intent-Based-Chatbot.git
cd Intent-Based-Chatbot

# Install dependencies
pip install -r requirements.txt

# (Optional) Re-train the model
python preprocessing.py
python model.py

# Run the chatbot
python chat.py
