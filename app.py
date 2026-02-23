from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("food_spoilage_model.pkl")

@app.route("/")
def home():
    return "Food Spoilage API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        temperature = float(data["temperature"])
        cookedTime = float(data["cookedTime"])

        input_data = np.array([[temperature, cookedTime]])

        prediction = model.predict(input_data)

        result = "Fresh" if prediction[0] == 0 else "Spoiled"

        # ✅ THIS IS IMPORTANT
        return jsonify({
            "result": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run()