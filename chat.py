import json
import numpy as np
from tensorflow import keras
import pickle
import colorama
from colorama import Fore, Style
import random

colorama.init()

# Load intents JSON
with open('intents.json') as file:
    data = json.load(file)

# Load saved model and preprocessing objects
model = keras.models.load_model('chat_model')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

max_len = 20

def chat():
    print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
    while True:
        inp = input(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL)
        if inp.lower() == "quit":
            break

        sequences = tokenizer.texts_to_sequences([inp])
        padded = keras.preprocessing.sequence.pad_sequences(sequences, truncating='post', maxlen=max_len)
        predictions = model.predict(padded)
        tag = lbl_encoder.inverse_transform([np.argmax(predictions)])

        for intent in data['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                print(Fore.GREEN + "ChatBot: " + Style.RESET_ALL + response)

if __name__ == "__main__":
    chat()
