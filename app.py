from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load("health_risk_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get JSON data from frontend
    print("Received Data:", data)  # Debugging

    if not data:
        return jsonify({"error": "No data received"}), 400

    age = data.get("age")
    lifestyle = data.get("lifestyle")
    conditions = data.get("pre_existing_conditions")

    if age is None or lifestyle is None or conditions is None:
        return jsonify({"error": "Missing fields"}), 400

    # Print available categories for debugging
    print("Available Lifestyles:", label_encoders["Lifestyle"].classes_)
    print("Available Conditions:", label_encoders["Pre-existing Conditions"].classes_)

    # Encode categorical features
    try:
        lifestyle_encoded = label_encoders["Lifestyle"].transform([lifestyle])[0]
        conditions_encoded = label_encoders["Pre-existing Conditions"].transform([conditions])[0]
    except ValueError as e:
        print("Encoding Error:", e)  # Debugging
        return jsonify({"error": "Invalid categorical value"}), 400

    # Prepare input for prediction
    features = np.array([[age, lifestyle_encoded, conditions_encoded]])
    prediction = model.predict(features)[0]

    # Decode prediction
    risk_label = label_encoders["Health Risk"].inverse_transform([prediction])[0]

    return jsonify({"health_risk": risk_label})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
