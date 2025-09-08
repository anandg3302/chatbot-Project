import os
import json
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Load trained model
# ----------------------------
model_path = None
if os.path.exists("chat_model.keras"):
    model_path = "chat_model.keras"
elif os.path.exists("chat_model.h5"):
    model_path = "chat_model.h5"

if not model_path:
    raise FileNotFoundError("❌ No trained model found. Run train_chatbot.py first.")

model = tf.keras.models.load_model(model_path)
print(f"✅ Loaded model from: {model_path}")

# ----------------------------
# Load artifacts
# ----------------------------
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("label_encoder.pickle", "rb") as enc_file:
    lbl_encoder = pickle.load(enc_file)

with open("responses.pickle", "rb") as resp_file:
    responses = pickle.load(resp_file)

# Parameters (must match training script)
max_len = 25  

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__, template_folder="templates")

def chatbot_response(user_input):
    """Generate chatbot response for user input."""
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, truncating="post", maxlen=max_len)

    prediction = model.predict(padded, verbose=0)
    intent_index = np.argmax(prediction)
    intent_tag = lbl_encoder.inverse_transform([intent_index])[0]

    bot_reply = np.random.choice(responses[intent_tag])
    return bot_reply

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("chat.html")   # frontend

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    bot_reply = chatbot_response(user_message)
    return jsonify({"reply": bot_reply})

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
