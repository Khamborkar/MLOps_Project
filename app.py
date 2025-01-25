from flask import Flask, request, jsonify
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from model import load_model_and_tokenizer, predict_sentiment
from model import preprocess_text


# Initialize Flask app
app = Flask(__name__)

@app.route('/api/sentiment', methods=['POST'])
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
