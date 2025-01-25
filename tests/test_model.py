import unittest
from src.model import model  # Replace with the actual function name
from src.model import clean_text
from src.model import remove_stop_words
from src.model import preprocess_text


class TestModel(unittest.TestCase):

    def test_model_output_shape(self):
        # Generate sample input data (replace with appropriate data for your model)
        # create 'processed_text' which contains the preprocessed text
        train_df = pd.read_csv("MLOps_Project\\Tweets.csv")
        train_df['text'] = train_df['text'].apply(clean_text)
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        train_df['text'] = train_df['text'].apply(preprocess_text)
        text = " ".join(train_df["text"])
        df['processed_text'] = df['text'].apply(clean_text)
        df['processed_text'] = df['processed_text'].apply(remove_stop_words)
        # Create sentiment value from sentiment
        df['airline_sentiment_value'] = df['airline_sentiment'].map({'positive': 1 , 'negative': 0 ,'neutral':2})
        df.head()
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
        # Get the model output
        model_output = model()
        test_loss, test_accuracy = model.evaluate(X_val,val_labels)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        return test_accuracy


if __name__ == '__main__':
    unittest.main()
