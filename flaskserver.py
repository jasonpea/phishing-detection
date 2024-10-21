from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load your trained phishing detection model
with open('phishing_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to adjust features
def adjust_new_url_features(new_url_features, expected_num_features):
    while len(new_url_features) < expected_num_features:
        new_url_features.append(0)
    return new_url_features

# Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url_features = data.get('features', [])
    
    if not url_features:
        return jsonify({"error": "No URL features provided"}), 400

    # Adjust the features and make predictions
    adjusted_features = np.array(adjust_new_url_features(url_features, 98)).reshape(1, -1)
    prediction = model.predict(adjusted_features)

    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
