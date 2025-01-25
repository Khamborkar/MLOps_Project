import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Embedding, Dense
# from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import string
from tensorflow.keras.callbacks import EarlyStopping
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
df = pd.read_csv("MLOps_Project\\Tweets.csv")
df.head()
train_df = pd.read_csv("MLOps_Project\\Tweets.csv")
# train_df.head()
# data validation
valid_df = pd.read_csv("MLOps_Project\\Tweets.csv")
# valid_df.head()
# data cleaning
train_df.shape
# train_df.isna().sum()
# train_df.duplicated().sum()
# train_df.shape


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
# plt.figure(figsize=(8, 6))
# sns.set_style(style="whitegrid")
# ax = sns.countplot(data=train_df, x='airline_sentiment', palette="Set2", edgecolor="black")
# plt.title(f'Count Plot of Sentiments', fontweight='bold', color='darkblue')
# plt.ylabel("Count", fontsize=12)
# for p in ax.patches:
#         ax.annotate(f'{int(p.get_height())}',
#                     (p.get_x() + p.get_width() / 2., p.get_height()),
#                     ha='center', va='bottom',
#                     xytext=(0, 5), textcoords='offset points',
#                     fontsize=10, color='black')
# plt.tight_layout(pad=2)
# plt.show()
# plt.figure(figsize=(6,6))
# plt.pie(x =train_df['airline_sentiment'].value_counts().values,
#             labels=train_df['airline_sentiment'].value_counts().keys(),
#             autopct="%1.1f%%", textprops={"fontsize":10,"fontweight":"black"})
# plt.title('Sentiments Distribution')
# plt.show()
pd.crosstab(train_df['airline_sentiment'], 
            train_df['airline_sentiment']).T.style.background_gradient(subset=['negative'],
                                                                       cmap='Reds')\.background_gradient(
                                                                                                         subset=['positive'], 
                                                                                                         cmap='Greens')\.background_gradient(subset=['neutral'], 
                                                                                                                                             cmap='Blues')
# Combine all text into one string
text = " ".join(train_df["text"])
tokens = text.split()
# Count the frequency of occurrence of each word
word_counts = Counter(tokens)
# Take the word with the highest
# frequency of occurrence
top_words = word_counts.most_common(20)
# Take the top 20 words
# --- Visualisasi 1: Bar Chart ---
# Separating words and their numbers
words, counts = zip(*top_words)
# # Setup seaborn theme
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(12, 6))
# sns.barplot(x=list(counts), y=list(words), palette="viridis")
# plt.title("20 Most Frequently Occurring Words", fontsize=16, fontweight='bold')
# plt.xlabel("Frequency", fontsize=14)
# plt.ylabel("Words", fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis'
).generate_from_frequencies(word_counts)
# # WordCloud
# plt.figure(figsize=(12, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title("WordCloud of Frequently Appearing Words",
# fontsize=16, fontweight='bold')
# plt.show()
positive_words = ' '.join(positive_sentiment)
# Tokenize the words and
# count frequency for positive sentiment
words_positif = positive_words.split()
word_counts_positif = Counter(words_positif)
# Display the most common words for positive sentiment
common_words_positif = word_counts_positif.most_common(10)
# Create word cloud for positive
# sentiment with a green colormap
wordcloud_positif = WordCloud(
    width=800, height=400, background_color='white',
    colormap='Greens').generate_from_frequencies(word_counts_positif)
# # Plotting the word cloud for positive sentiment
# plt.figure(figsize=(10, 6))
# plt.imshow(wordcloud_positif, interpolation='bilinear')
# plt.axis('off')
# plt.title("Word Cloud for Positive Sentiment (Green)")
# plt.show()
neutral_words = ' '.join(neutral_sentiment)
# Tokenize the words and count frequency for netral sentiment
words_netral = neutral_words.split()
word_counts_netral = Counter(words_netral)
# Display the most common words for netral sentiment
common_words_netral = word_counts_netral.most_common(10)
# Create word cloud for netral sentiment with a red colormap
wordcloud_netral = WordCloud(
    width=800, height=400, background_color='white',
    colormap='Blues').generate_from_frequencies(word_counts_netral)
# # Plotting the word cloud for netral sentiment
# plt.figure(figsize=(10, 6))
# plt.imshow(wordcloud_netral, interpolation='bilinear')
# plt.axis('off')
# plt.title("Word Cloud for Neutral Sentiment (Blue)")
# plt.show()
# Join all the positive and negative reviews into single strings
negative_words = ' '.join(negative_sentiment)
# Tokenize the words and count frequency for negative sentiment
words_negatif = negative_words.split()
word_counts_negatif = Counter(words_negatif)
# Display the most common words for negative sentiment
common_words_negatif = word_counts_negatif.most_common(10)
# Create word cloud for negative sentiment with a red colormap
wordcloud_negatif = WordCloud(
    width=800, height=400,
    background_color='white',
    colormap='Reds').generate_from_frequencies(word_counts_negatif)
# # Plotting the word cloud for negative sentiment
# plt.figure(figsize=(10, 6))
# plt.imshow(wordcloud_negatif, interpolation='bilinear')
# plt.axis('off')
# plt.title("Word Cloud for Negative Sentiment (Red)")
# plt.show()
# Feature Extraction
train_df =train_df[['text','airline_sentiment']]
train_df.sample(5)
valid_df =valid_df[['text','airline_sentiment']]
valid_df.sample(5)
train_df['airline_sentiment'] = train_df['airline_sentiment'].map({'positive': 1 ,
                                                                   'negative': 0 ,'neutral':2 , 'irrelevant': 2})
valid_df['airline_sentiment'] = valid_df['airline_sentiment'].map({'positive': 1 ,
                                                                   'negative': 0 ,'neutral':2 , 'irrelevant': 2})
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

# Function to remove the stop words
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words and word not in string.punctuation]
    return " ".join(filtered_words)
    
# create 'processed_text' which contains the preprocessed text
df['processed_text'] = df['text'].apply(clean_text)
df['processed_text'] = df['processed_text'].apply(remove_stop_words)
# Create sentiment value from sentiment
df['airline_sentiment_value'] = df['airline_sentiment'].map({'positive': 1 , 'negative': 0 , 'neutral':2})
# df.head()
X_train, X_temp, y_train, y_temp = train_test_split(df['processed_text'], df['airline_sentiment_value'], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# Tokenization and padding
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)
val_sequences = tokenizer.texts_to_sequences(X_val)
max_sequence_length = 100
X_train = pad_sequences(train_sequences, maxlen=max_sequence_length)
X_test = pad_sequences(test_sequences, maxlen=max_sequence_length)
X_val = pad_sequences(val_sequences, maxlen=max_sequence_length)
# Convert labels to NumPy arrays
train_labels = np.array(y_train)
test_labels = np.array(y_test)
val_labels = np.array(y_val)
# Create the LSTM model and compile
model = models.Sequential([
    Embedding(input_dim=10000, output_dim=100,
              input_length=max_sequence_length),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dropout(0.2),
    Dense(3, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_summary = model.summary()
# Fit the model with the training data
# Using early stopping because the results was not converging for test data
# Create an EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=3, restore_best_weights=True)
# Fit the model with the EarlyStopping callback
# Reshape the input sequences to include the sequence length dimension
    # classifier.train_sequences = classifier.train_sequences.reshape(
    #     classifier.train_sequences.shape[0], 1, classifier.train_sequences.shape[1]
    # )
    # classifier.val_sequences = classifier.val_sequences.reshape(
    #     classifier.val_sequences.shape[0], 1, classifier.val_sequences.shape[1]
    # )
history = model.fit(
    X_train, train_labels,
    validation_data=(X_val, val_labels),
    epochs=20,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping]
)
model.save('model.h5')
test_loss, test_accuracy = model.evaluate(X_test, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
