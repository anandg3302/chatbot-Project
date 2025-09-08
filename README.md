🤖 Chatbot Project

A simple AI-powered chatbot built with TensorFlow/Keras and served via a Flask API.
It responds to user queries based on predefined intents stored in intents.json.

📸 Demo

(You can add a screenshot or GIF here once your chatbot is running.)

📂 Project Structure
chatbot-project/
│
├── intents.json              # Training data (intents, patterns, responses)
├── train_chatbot.py          # Script to train the chatbot model
├── app.py                    # Flask API to serve chatbot responses
├── chat.html                 # Simple frontend for chatting
├── tokenizer.pickle          # Saved tokenizer (generated after training)
├── label_encoder.pickle      # Saved label encoder (generated after training)
├── responses.pickle          # Saved responses dictionary
├── chat_model.keras          # Trained chatbot model (new format)
├── chat_model.h5             # Trained chatbot model (legacy format)
└── venv/                     # Virtual environment (not pushed to GitHub)

⚙️ Installation

Clone the repository:

git clone https://github.com/anandg3302/chatbot-Project.git
cd chatbot-Project


Create & activate a virtual environment:

python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

🏋️ Train the Model

Run the training script:

python train_chatbot.py


This will generate:

chat_model.keras (trained model)

tokenizer.pickle

label_encoder.pickle

responses.pickle

🚀 Run the Flask API

Start the backend server:

python app.py


The server will start at:
👉 http://127.0.0.1:5000/chat

Test with curl:

curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d "{\"message\":\"hello\"}"

💻 Frontend

Open chat.html in your browser.
It provides a simple UI to interact with your chatbot.

🗂 Example Intents

intents.json defines training data:

{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey"],
      "responses": ["Hello!", "Hi there!", "Hey! How can I help?"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later"],
      "responses": ["Goodbye!", "See you soon!"]
    }
  ]
}

🛠 Tech Stack

Python 3.11+

TensorFlow / Keras (Deep Learning)

Flask (Backend API)

scikit-learn (Label encoding)

HTML/CSS/JavaScript (Frontend)

🚧 Future Improvements

Improve NLP with transformers (e.g., BERT, GPT-based models).

Add user authentication.

Deploy chatbot to Heroku, AWS, or Docker.

Add context handling for multi-turn conversations.

📜 License

This project is open-source under the MIT License.
