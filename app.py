from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

model_path = os.path.join(os.getcwd(), "food_spoilage_model.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return "Food Spoilage API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    temperature = data["temperature"]
    cooked_time = data["cooked_time"]

    # 🔥 THIS IS THE IMPORTANT FIX
    input_data = pd.DataFrame(
        [[temperature, cooked_time]],
        columns=["temperature", "cooked_time"]
    )

    prediction = model.predict(input_data)

    return jsonify({
        "spoilage": int(prediction[0])
    })