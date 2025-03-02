import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

# Load dataset using TensorFlow Datasets (TFDS)
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_data, test_data = dataset['train'], dataset['test']

# Hyperparameters
vocab_size = 1000
embedding_dim = 16
max_length = 10
oov_token = "<OOV>"

# Tokenize
# Create vocabulary (word-to-index)
word_counts = {}
for sentence, _ in train_data:
    for word in sentence.numpy().decode('utf-8').lower().split():
        word_counts[word] = word_counts.get(word, 0) + 1

# Sort words by frequency and take the top vocab_size words
sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
vocab = {word: idx+1 for idx, word in enumerate(sorted_words[:vocab_size-1])}
vocab[oov_token] = 0  # Assign OOV token to index 0

# Convert sentences to sequences (list of word indices)
def text_to_sequence(text):
    text = text.numpy().decode('utf-8').lower()
    return [vocab.get(word, vocab[oov_token]) for word in text.split()]

# Convert dataset to padded sequences
def preprocess_data(dataset):
    sequences = []
    labels = []
    for sentence, label in dataset:
        seq = text_to_sequence(sentence)
        sequences.append(seq)
        labels.append(label.numpy())
    return sequences, np.array(labels)

# Preprocess train and test data
train_sequences, train_labels = preprocess_data(train_data)
test_sequences, test_labels = preprocess_data(test_data)

# Pad the sequences so they all have the same length
train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Define the model structure
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(32),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compiles model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Trains model
model.fit(train_sequences, train_labels, epochs=15, batch_size=24, verbose=1)

# Saves model
model.save("sentiment_analysis_bot.keras")

# Performs an eval on the model
test_loss, test_accuracy = model.evaluate(test_sequences, test_labels)
print(f'Test accuracy: {test_accuracy:.4f}')

# Function to streamline the process of using the model
def predict_sentiment(text):
    seq = text_to_sequence(tf.constant(text))  # Convert text to sequence
    padded = pad_sequences([seq], maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"

# Usage
print(predict_sentiment("I really enjoyed the movie, it was amazing!"))
print(predict_sentiment("This was the worst film I've ever seen."))
