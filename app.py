from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model safely
MODEL_PATH = "food_spoilage_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found!")

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return "Food Spoilage API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON received"}), 400

        # Get values from Android
        temperature = float(data.get("temperature"))
        cooked_time = float(data.get("cooked_time"))

        # IMPORTANT: model was trained using column name "temp"
        input_df = pd.DataFrame(
            [[temperature, cooked_time]],
            columns=["temp", "cooked_time"]
        )

        prediction = model.predict(input_df)

        result = "Fresh" if int(prediction[0]) == 0 else "Spoiled"

        return jsonify({
            "result": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)