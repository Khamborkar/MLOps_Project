from flask import Flask, request, jsonify
from model import load_model_and_tokenizer, predict_sentiment

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer during app initialization
model, tokenizer = load_model_and_tokenizer()

@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        # Get input text from the request
        input_data = request.get_json()
        raw_text = input_data['text']

        # Predict sentiment
        prediction = predict_sentiment(model, tokenizer, raw_text)
        
        # Interpret the prediction
        sentiment_label = ['Negative', 'Positive', 'Neutral']
        sentiment = sentiment_label[prediction.argmax()]
        confidence = round(float(prediction.max()), 2)

        # Return response
        return jsonify({
            "input_text": raw_text,
            "predicted_sentiment": sentiment,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the Flask application
    app.run(host='0.0.0.0', port=8000)
