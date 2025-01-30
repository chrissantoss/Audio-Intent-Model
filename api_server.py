from flask import Flask, request, jsonify
from flask_cors import CORS
from intent_classifier import IntentClassifier
import os

app = Flask(__name__)
CORS(app)
classifier = IntentClassifier()

@app.route('/')
def home():
    return "Intent Recognition API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received request data: {data}")
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        intent, confidence = classifier.predict(text)
        
        response = {
            'intent': intent,
            'confidence': float(confidence)
        }
        print(f"Sending response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    ) 