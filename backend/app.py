from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://192.168.1.4:3000"])

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return "Diabetes Prediction Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [
        float(data["Pregnancies"]),
        float(data["Glucose"]),
        float(data["BloodPressure"]),
        float(data["SkinThickness"]),
        float(data["Insulin"]),
        float(data["BMI"]),
        float(data["DiabetesPedigreeFunction"]),
        float(data["Age"])
    ]

    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)