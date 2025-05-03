from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "clr.pkl")
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None

# Home route
@app.route("/")
def home():
    return "Telecom Customer Predictor is Live!"

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded properly"}), 500

    try:
        data = request.get_json(force=True)

        expected_features = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges", "Total_Main_Servies",
            "Total_Secondry_Servies"
        ]

        missing = [f for f in expected_features if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        input_data = [data[feature] for feature in expected_features]
        input_array = np.array([input_data])

        prediction = model.predict(input_array)
        return jsonify({"prediction": float(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
