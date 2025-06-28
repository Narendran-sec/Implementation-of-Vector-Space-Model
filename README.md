# Implementation-of-Vector-Space-Model

## Aim:
To design and implement a Siamese neural network using TensorFlow and Keras to measure the semantic similarity between pairs of sentences.

## Requirements:
**Software Requirements:**
Python 3.x
TensorFlow 2.x
NumPy
Libraries Used:
tensorflow
numpy**

**Hardware Requirements:**
Basic machine with CPU or GPU (GPU recommended for faster training)

## Procedure:
Data Preparation:

Define pairs of sentences with similarity labels (1 for similar, 0 for dissimilar).

Tokenize and pad the sentences to ensure uniform input length.

Model Architecture:

Use an Embedding layer to convert words into vector representations.

Use a shared Bidirectional LSTM layer to encode both sentences.

Compute the absolute difference (L1 distance) between encoded vectors.

Pass the result to a Dense layer with sigmoid activation to predict similarity.

Training:

Train the model on sentence pairs using binary cross-entropy loss and the Adam optimizer.

Run training for a number of epochs with a small batch size.

Evaluation:

Predict similarity on new sentence pairs and display the similarity score.

## Program:
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

# Sentence pairs and labels
sentence_pairs = [
    ("How are you?", "How do you do?", 1),
    ("How are you?", "What is your name?", 0),
    ("What time is it?", "Can you tell me the time?", 1),
    ("What is your name?", "Tell me the time?", 0),
    ("Hello there!", "Hi!", 1),
]

sentences1 = [pair[0] for pair in sentence_pairs]
sentences2 = [pair[1] for pair in sentence_pairs]
labels = np.array([pair[2] for pair in sentence_pairs])

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences1 + sentences2)
vocab_size = len(tokenizer.word_index) + 1

max_len = 100
X1 = pad_sequences(tokenizer.texts_to_sequences(sentences1), maxlen=max_len)
X2 = pad_sequences(tokenizer.texts_to_sequences(sentences2), maxlen=max_len)

# Input layers
input_1 = Input(shape=(max_len,))
input_2 = Input(shape=(max_len,))

# Embedding and shared LSTM
embedding_dim = 1000
embedding = Embedding(vocab_size, embedding_dim, input_length=max_len)
shared_lstm = Bidirectional(LSTM(512))

encoded_1 = shared_lstm(embedding(input_1))
encoded_2 = shared_lstm(embedding(input_2))

# L1 distance
def l1_distance(vectors):
    x, y = vectors
    return K.abs(x - y)

l1_layer = Lambda(l1_distance)
l1_distance_output = l1_layer([encoded_1, encoded_2])

# Output layer
output = Dense(1, activation='sigmoid')(l1_distance_output)

# Model creation
siamese_network = Model([input_1, input_2], output)
siamese_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
siamese_network.summary()

# Train the model
siamese_network.fit([X1, X2], labels, epochs=12, batch_size=2)

# Test data
test_sentences1 = ["How are you?"]
test_sentences2 = ["How do you do?"]

test_X1 = pad_sequences(tokenizer.texts_to_sequences(test_sentences1), maxlen=max_len)
test_X2 = pad_sequences(tokenizer.texts_to_sequences(test_sentences2), maxlen=max_len)

# Prediction
similarity = siamese_network.predict([test_X1, test_X2])
print(f"Similarity Score: {similarity[0][0]}")
```
## Output:
```python-repl

Epoch 1/12
...
Similarity Score: 0.89
(Note: The exact score may vary each time due to random initialization.)
```
## Result:
The Siamese neural network successfully learned to predict sentence similarity, showing high similarity for semantically related sentence pairs.
