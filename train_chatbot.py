import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Suppress TensorFlow logs except errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ----------------------------
# Load intents JSON file
# ----------------------------
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = {}

# Extract patterns, tags, and responses
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        training_sentences.append(pattern)
        training_labels.append(intent["tag"])

    responses[intent["tag"]] = intent["responses"]

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

num_classes = len(labels)

# ----------------------------
# Encode labels into numbers
# ----------------------------
lbl_encoder = LabelEncoder()
training_labels_encoded = lbl_encoder.fit_transform(training_labels)

# ----------------------------
# Tokenize text data
# ----------------------------
vocab_size = 2000
embedding_dim = 32
max_len = 25
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating="post", maxlen=max_len)

# ----------------------------
# Define the model architecture
# ----------------------------
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print(model.summary())

# ----------------------------
# Train the model with early stopping
# ----------------------------
epochs = 200
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=10,
    restore_best_weights=True
)

model.fit(
    padded_sequences,
    np.array(training_labels_encoded),
    epochs=epochs,
    callbacks=[early_stop],
    verbose=1
)

# ----------------------------
# Save model + artifacts
# ----------------------------
# Save in both formats for flexibility
model.save("chat_model.keras", save_format="keras")   # ✅ modern format
model.save("chat_model.h5")                           # ✅ legacy format

with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("label_encoder.pickle", "wb") as enc_file:
    pickle.dump(lbl_encoder, enc_file, protocol=pickle.HIGHEST_PROTOCOL)

with open("responses.pickle", "wb") as resp_file:
    pickle.dump(responses, resp_file, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Model and training artifacts saved successfully!")
