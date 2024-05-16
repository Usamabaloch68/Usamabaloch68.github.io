from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import random
import json
import pickle
import numpy as np
import nltk

app = Flask(__name__)
CORS(app)
messages = []

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('D:\BOT\Project\Project\intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('D:\BOT\Project\Project\chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

print("GO! Bot is running!")

@app.route("/messages", methods=["GET"])
def get_messages():
    return jsonify(messages)

@app.route("/process_input", methods=["POST"])
def process_input():
    user_input = request.form["input_text"]
    messages.append({"user": True, "content": user_input})
    
    ints = predict_class (user_input)  # Replace with your AI logic
    ai_response = get_response (ints, intents)
    messages.append({"user": False, "content": ai_response})
    return jsonify(messages)


@app.route("/clear_input", methods=["GET"])
def clear_input():
    global messages  
    messages = []    
    return jsonify(messages)




def generate_ai_response(user_input):
    return user_input

if __name__ == "__main__":
    app.run(debug=True)
