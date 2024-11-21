import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load your dataset
dataset_path = "IMDB Dataset.csv"  # Replace with the actual path to your dataset
data = pd.read_csv(dataset_path)

# Encode sentiments: positive -> 1, negative -> 0
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)

# Tokenization and padding
max_features = 10000  # Vocabulary size
maxlen = 200          # Maximum sequence length

tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post')

# Build the model
model = Sequential([
    Embedding(max_features, 128),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 5
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Save the model
model.save('sentiment_analysis_model.keras')

# Predict sentiment for a custom review
def predict_review(review, model, tokenizer):
    encoded = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(encoded, maxlen=maxlen, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"

# Example usage
custom_review = "The movie was breathtaking and absolutely stunning!"
print(f"Sentiment: {predict_review(custom_review, model, tokenizer)}")