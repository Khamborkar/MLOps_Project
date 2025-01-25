import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model
import joblib

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


stemmer = PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))


def preprocess_text(text):
    words = text.split()
    processed_words = [
        stemmer.stem(word) for word in words if word not in stop_words
    ]
    return ' '.join(processed_words)


def build_model():
    model = models.Sequential([
        Embedding(
            input_dim=10000,
            output_dim=128,
            input_length=100
        ),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Load the trained model and tokenizer
def load_model_and_tokenizer():
    # model = load_model('model.h5')
    model = build_model()
    test_model = model.save(test_model.h5)
    tokenizer = joblib.load('tokenizer.pkl')
    return model, tokenizer


# Predict sentiment
def predict_sentiment(model, tokenizer, text, max_len=100):
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    return prediction


if __name__ == "__main__":
    df = pd.read_csv("MLOps_Project/Tweets.csv")
    df['processed_text'] = df['text'].apply(clean_text).apply(preprocess_text)

    df['airline_sentiment_value'] = df['airline_sentiment'].map({
        'positive': 1,
        'negative': 0,
        'neutral': 2
    })

    X_train, X_temp, y_train, y_temp = train_test_split(
        df['processed_text'],
        df['airline_sentiment_value'],
        test_size=0.3,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)

    max_sequence_length = 100
    X_train = pad_sequences(
        tokenizer.texts_to_sequences(X_train),
        maxlen=max_sequence_length
    )
    X_val = pad_sequences(
        tokenizer.texts_to_sequences(X_val),
        maxlen=max_sequence_length
    )
    X_test = pad_sequences(
        tokenizer.texts_to_sequences(X_test),
        maxlen=max_sequence_length
    )

    train_labels = np.array(y_train)
    val_labels = np.array(y_val)
    test_labels = np.array(y_test)

    model = build_model()
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )

    history = model.fit(
        X_train, train_labels,
        validation_data=(X_val, val_labels),
        epochs=20, batch_size=32, callbacks=[early_stopping]
    )
    model.save('model.h5')
    test_loss, test_accuracy = model.evaluate(X_test, test_labels)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
