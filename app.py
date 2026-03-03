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
    try:
        data = request.get_json()

        temp = float(data["temp"])
        humidity = float(data["humidity"])
        cooked_time = float(data["cooked_time"])

        input_data = pd.DataFrame(
            [[temp, humidity, cooked_time]],
            columns=["temp", "humidity", "cooked_time"]
        )

        prediction = model.predict(input_data)

        if int(prediction[0]) == 0:
            result = "Fresh"
        else:
            result = "Spoiled"

        return jsonify({
            "spoilage": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)