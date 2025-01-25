import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.model import model, clean_text, remove_stop_words, preprocess_text


class TestModel(unittest.TestCase):
    """Unit test class for the model."""

    def test_model_output_shape(self):
        """Test to validate the model's output accuracy."""
        # Load dataset
        train_df = pd.read_csv("MLOps_Project\\Tweets.csv")
        train_df['text'] = train_df['text'].apply(clean_text)
        train_df['text'] = train_df['text'].apply(preprocess_text)

        # Processed text
        train_df['processed_text'] = train_df['text'].apply(remove_stop_words)

        # Create sentiment mapping
        dummy_df = train_df['airline_sentiment'].map({
                                                        'positive': 1,
                                                        'negative': 0,
                                                        'neutral': 2
                                                    })
        train_df['airline_sentiment_value'] = dummy_df
        # Train/test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            train_df['processed_text'],
            train_df['airline_sentiment_value'],
            test_size=0.3,
            random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=42
        )

        # Tokenization and padding
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(X_train)

        X_train = pad_sequences(
            tokenizer.texts_to_sequences(X_train),
            maxlen=100
        )
        X_val = pad_sequences(
            tokenizer.texts_to_sequences(X_val),
            maxlen=100
        )
        X_test = pad_sequences(
            tokenizer.texts_to_sequences(X_test),
            maxlen=100
        )

        # Convert labels to NumPy arrays
        val_labels = np.array(y_val)

        # Load the model and evaluate
        model_instance = model()
        test_loss, test_accuracy = model_instance[0].evaluate(
            X_val,
            val_labels
        )

        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        self.assertGreater(
            test_accuracy, 0.5, "Model accuracy is below acceptable threshold."
        )


if __name__ == '__main__':
    unittest.main()
