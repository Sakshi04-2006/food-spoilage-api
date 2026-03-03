from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("food_model.pkl")

@app.route("/")
def home():
    return "Food Spoilage API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        # Extract values (MUST match training features)
        temp = float(data.get("temp", 0))
        humidity = float(data.get("humidity", 0))
        cooked_time = float(data.get("cooked_time", 0))

        # Create dataframe with EXACT feature names
        input_df = pd.DataFrame(
            [[temp, humidity, cooked_time]],
            columns=["temp", "humidity", "cooked_time"]
        )

        # Make prediction
        prediction = model.predict(input_df)

        result = prediction[0]

        return jsonify({"result": str(result)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)