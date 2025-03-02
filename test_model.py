import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

model = tf.keras.models.load_model("sentiment_analysis_bot.keras")

vocab_size = 1000
embedding_dim = 16
max_length = 10
oov_token = "<OOV>"

dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_data, test_data = dataset['train'], dataset['test']

word_counts = {}
for sentence, _ in train_data:
    for word in sentence.numpy().decode('utf-8').lower().split():
        word_counts[word] = word_counts.get(word, 0) + 1

sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
vocab = {word: idx+1 for idx, word in enumerate(sorted_words[:vocab_size-1])}
vocab[oov_token] = 0


def text_to_sequence(text):
    text = text.numpy().decode('utf-8').lower()
    return [vocab.get(word, vocab[oov_token]) for word in text.split()]

def predict_sentiment(text):
    seq = text_to_sequence(tf.constant(text))  # Convert text to sequence
    padded = pad_sequences([seq], maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    return f"Positive {prediction}" if prediction > 0.5 else f"Negative {prediction}"

while True:
    print(predict_sentiment(input("Try the sentient bot: ")))