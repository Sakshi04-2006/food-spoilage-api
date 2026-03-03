from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Food Spoilage API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)

        temp = float(data["temp"])
        humidity = float(data["humidity"])
        cooked_time = float(data["cooked_time"])

        print("Inputs:", temp, humidity, cooked_time)

        input_df = pd.DataFrame(
            [[temp, humidity, cooked_time]],
            columns=["temp", "humidity", "cooked_time"]
        )

        print("Model expects:", model.feature_names_in_)

        prediction = model.predict(input_df)

        result = "Fresh" if int(prediction[0]) == 0 else "Spoiled"

        return jsonify({"result": result})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500