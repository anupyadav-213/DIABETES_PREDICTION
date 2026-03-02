from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Diabetes Prediction Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ===== Original 8 features =====
        Pregnancies = float(data["Pregnancies"])
        Glucose = float(data["Glucose"])
        BloodPressure = float(data["BloodPressure"])
        SkinThickness = float(data["SkinThickness"])
        Insulin = float(data["Insulin"])
        BMI = float(data["BMI"])
        DiabetesPedigreeFunction = float(data["DiabetesPedigreeFunction"])
        Age = float(data["Age"])

        # ===== SAME Feature Engineering AS TRAINING =====
        Glucose_BMI = Glucose * BMI
        Age_BMI = Age * BMI
        Glucose_Insulin_ratio = Glucose / (Insulin + 1)
        BMI_category = 0
        if BMI < 18.5:
            BMI_category = 0
        elif BMI < 25:
            BMI_category = 1
        elif BMI < 30:
            BMI_category = 2
        else:
            BMI_category = 3

        Age_group = 0
        if Age < 30:
            Age_group = 0
        elif Age < 45:
            Age_group = 1
        elif Age < 60:
            Age_group = 2
        else:
            Age_group = 3

        High_glucose = 1 if Glucose > 140 else 0

        # ===== Final 14 feature array (IMPORTANT ORDER) =====
        features = [[
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age,
            Glucose_BMI,
            Age_BMI,
            Glucose_Insulin_ratio,
            BMI_category,
            Age_group,
            High_glucose
        ]]

        # Scale
        scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled)[0]

        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return jsonify({"prediction": result})

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)