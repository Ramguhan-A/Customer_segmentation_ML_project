from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained Decision Tree model
with open("model_decision_tree.pkl", "rb") as obj:
    model = pickle.load(obj)

# Define expected feature names
FEATURE_NAMES = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT",
    "TENURE"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Getting JSON input from API request

    # Extract feature values in the correct order
    features = np.array([data[col] for col in FEATURE_NAMES]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)

''' input: (changes and test)
FEATURE_NAMES = {
    "BALANCE": 7000,
    "BALANCE_FREQUENCY": 0.9,
    "PURCHASES": 500,
    "ONEOFF_PURCHASES": 200,
    "INSTALLMENTS_PURCHASES": 300,
    "CASH_ADVANCE": 1000,
    "PURCHASES_FREQUENCY": 0.8,
    "ONEOFF_PURCHASES_FREQUENCY": 0.6,
    "PURCHASES_INSTALLMENTS_FREQUENCY": 0.7,
    "CASH_ADVANCE_FREQUENCY": 0.5,
    "CASH_ADVANCE_TRX": 3,
    "PURCHASES_TRX": 10,
    "CREDIT_LIMIT": 5000,
    "PAYMENTS": 1200,
    "MINIMUM_PAYMENTS": 150,
    "PRC_FULL_PAYMENT": 0.4,
    "TENURE": 12
} '''