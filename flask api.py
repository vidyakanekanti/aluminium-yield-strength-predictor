from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from any origin

# Load model
with open("regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.route("/")
def home():
    return "Aluminum Yield Strength Prediction API is running! 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    processing_technique = data.get("Processing")
    tensile_strength = data.get("Tensile Strength (MPa)")

    if not processing_technique or tensile_strength is None:
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        processing_encoded = label_encoder.transform([processing_technique])[0]
    except ValueError:
        return jsonify({"error": "Processing technique not found in trained data"}), 400

    prediction = model.predict(np.array([[processing_encoded, tensile_strength]]))[0]
    return jsonify({"Predicted Yield Strength (MPa)": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)
