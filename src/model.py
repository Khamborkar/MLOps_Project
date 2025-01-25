# lint-fixme: NoInheritFromObject

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split
import string
from tensorflow.keras.callbacks import EarlyStopping
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
df = pd.read_csv("MLOps_Project\\Tweets.csv")
df.head()
train_df = pd.read_csv("MLOps_Project\\Tweets.csv")
# data validation
valid_df = pd.read_csv("MLOps_Project\\Tweets.csv")
# data cleaning
train_df.shape


def clean_text(text):
    # Convert all text to lowercase.
    text = text.lower()
    # Remove link/URL
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    # Remove emoji and non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove punctuation and other symbols
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Delete numbers (optional, if you don't want
    # to delete numbers, delete this line)
    text = re.sub(r'\d+', ' ', text)
    # Remove any double spaces that may form
    text = re.sub(r'\s+', ' ', text).strip()
    # Normalize Words
    text = re.sub(r"won\'t", "would not", text)
    text = re.sub(r"im", "i am", text)
    text = re.sub(r"Im", "i am", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"don\'t", "do not", text)
    text = re.sub(r"shouldn\'t", "should not", text)
    text = re.sub(r"needn\'t", "need not", text)
    text = re.sub(r"hasn\'t", "has not", text)
    text = re.sub(r"haven\'t", "have not", text)
    text = re.sub(r"weren\'t", "were not", text)
    text = re.sub(r"mightn\'t", "might not", text)
    text = re.sub(r"didn\'t", "did not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r'unk', ' ', text)
    return text


train_df['text'] = train_df['text'].apply(clean_text)
valid_df['text'] = valid_df['text'].apply(clean_text)
# train_df.head()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    words = text.split()
    processed_words = [stemmer.stem(word) for word in words
                       if word not in stop_words]
    return ' '.join(processed_words)


# Apply preprocessing to all text
train_df['text'] = train_df['text'].apply(preprocess_text)
valid_df['text'] = valid_df['text'].apply(preprocess_text)
# train_df.head()
positive_words = train_df[train_df['airline_sentiment'] == 'positive']
positive_sentiment = positive_words['text']
negative_words = train_df[train_df['airline_sentiment'] == 'negative']
negative_sentiment = negative_words['text']
neutral_words = train_df[train_df['airline_sentiment'] == 'neutral']
neutral_sentiment = neutral_words['text']
irrelevant_words = train_df[train_df['airline_sentiment'] == 'irrelevant']
irrelevant_sentiment = irrelevant_words['text']

# Combine all text into one string
text = " ".join(train_df["text"])
tokens = text.split()
word_counts = Counter(tokens)

# Feature Extraction
train_df = train_df[['text','airline_sentiment']]
train_df.sample(5)
valid_df = valid_df[['text','airline_sentiment']]
valid_df.sample(5)
train_df['airline_sentiment'] = train_df['airline_sentiment'].map(
    {
        'positive' : 1,
        'negative' : 0,
        'neutral' : 2,
        'irrelevant' : 2
    })
valid_df['airline_sentiment'] = valid_df['airline_sentiment'].map(
    {
        'positive' : 1,
        'negative' : 0,
        'neutral' : 2,
        'irrelevant' : 2
    })
train_texts = train_df['text'].values
train_labels = train_df['airline_sentiment'].values
val_texts = valid_df['text'].values
val_labels = valid_df['airline_sentiment'].values
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)
max_sequence_length = 100
X = pad_sequences(train_sequences, maxlen=max_sequence_length)
X_val = pad_sequences(val_sequences, maxlen=max_sequence_length)
with open('tokenizer2.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)
print("The tokenizer has been saved.")


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words and word not in string.punctuation]
    return " ".join(filtered_words)


# create 'processed_text' which contains the preprocessed text
df['processed_text'] = df['text'].apply(clean_text)
df['processed_text'] = df['processed_text'].apply(remove_stop_words)
# Create sentiment value from sentiment
df['airline_sentiment_value'] = df['airline_sentiment'].map(
    {
        'positive' : 1,
        'negative' : 0,
        'neutral' : 2
    })
# df.head()
X_train, X_temp, y_train, y_temp = train_test_split(df['processed_text'],
                                                    df['airline_sentiment_value'],
                                                    test_size = 0.3,
                                                    random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_temp,
                                                y_temp,
                                                test_size = 0.5,
                                                random_state = 42)
# Tokenization and padding
tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)
val_sequences = tokenizer.texts_to_sequences(X_val)
max_sequence_length = 100
X_train = pad_sequences(train_sequences,
                        maxlen = max_sequence_length)
X_test = pad_sequences(test_sequences,
                       maxlen = max_sequence_length)
X_val = pad_sequences(val_sequences,
                      maxlen = max_sequence_length)
# Convert labels to NumPy arrays
train_labels = np.array(y_train)
test_labels = np.array(y_test)
val_labels = np.array(y_val)


def model():
    model = models.Sequential([
        Embedding(input_dim = 10000,
                  output_dim = 100,
                  input_length = max_sequence_length),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dropout(0.2),
        Dense(3, activation = 'softmax')
    ])
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    # Create an EarlyStopping callback
    early_stopping = EarlyStopping(monitor = 'val_loss',
                                   patience = 3,
                                   restore_best_weights = True)
    history = model.fit(
        X_train, train_labels,
        validation_data = (X_val, val_labels),
        epochs = 20,
        batch_size = 32,
        verbose = 1,
        callbacks = [early_stopping]
    )
    model.save('model.h5')
    list = [model, history]
    return list


test_loss, test_accuracy = model()[0].evaluate(X_test, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
history = model()[1]
