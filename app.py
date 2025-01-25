from flask import Flask, request, jsonify
from model import load_model_and_tokenizer, predict_sentiment
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify


# Initialize Flask app
app = Flask(__name__)


@app.route('/api/sentiment', methods=['POST'])
# Function to load the model and tokenizer
def load_model_and_tokenizer():
    model = load_model('sentiment_model.h5')  # Load the Keras model
    tokenizer = joblib.load('tokenizer.pkl')  # Load the tokenizer using joblib
    return model, tokenizer


# Function to preprocess the input text
def preprocess_text(text):
    """
    Preprocess text by removing unwanted characters and making it lowercase.
    Modify this function based on your text preprocessing pipeline.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text


# Function to predict sentiment
def predict_sentiment(model, tokenizer, text, max_len=100):
    """
    Predict the sentiment of the input text.
    """
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    return prediction


# Load the model and tokenizer once at the start
model, tokenizer = load_model_and_tokenizer()


# Define a route to predict sentiment from text
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the request
    input_text = request.json.get('text')
    if not input_text:
        return jsonify({'error': 'No text provided'}), 400

    # Make a prediction
    prediction = predict_sentiment(model, tokenizer, input_text)

    # Convert prediction to a user-friendly format (optional)
    sentiment = 'positive' if prediction >= 0.5 else 'negative'

    # Return the result as JSON
    return jsonify({'sentiment': sentiment, 'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
