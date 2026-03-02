import React, { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    Pregnancies: "",
    Glucose: "",
    BloodPressure: "",
    SkinThickness: "",
    Insulin: "",
    BMI: "",
    DiabetesPedigreeFunction: "",
    Age: ""
  });

  const [result, setResult] = useState("");

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

const handleSubmit = async (e) => {
  e.preventDefault();

  try {
    const response = await fetch("https://diabetes-prediction-r3hz.onrender.com/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(formData)
    });

    const data = await response.json();
    console.log("Response from backend:", data);
    setResult(data.prediction);

  } catch (error) {
    console.error("Error:", error);
  }
};

  return (
    <div style={{ textAlign: "center", marginTop: "40px" }}>
      <h1>Diabetes Prediction</h1>

      <form onSubmit={handleSubmit} style={{ maxWidth: "400px", margin: "auto" }}>
        {Object.keys(formData).map((key) => (
          <div key={key} style={{ marginBottom: "10px" }}>
            <input
              type="number"
              name={key}
              placeholder={key}
              value={formData[key]}
              onChange={handleChange}
              required
              style={{ width: "100%", padding: "8px" }}
            />
          </div>
        ))}

        <button type="submit" style={{ padding: "10px 20px" }}>
          Predict
        </button>
      </form>

      {result && (
        <h2 style={{ marginTop: "20px" }}>
          Prediction: {result}
        </h2>
      )}
    </div>
  );
}

export default App;