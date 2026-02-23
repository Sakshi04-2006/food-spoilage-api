from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("food_spoilage_model.pkl")

@app.route("/")
def home():
    return "Food Spoilage API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        temperature = float(data.get("temperature", 0))
        cooked_time = float(data.get("cooked_time", 0))

        input_data = np.array([[temperature, cooked_time]])

        prediction = model.predict(input_data)

        if int(prediction[0]) == 0:
            result = "Fresh"
        else:
            result = "Spoiled"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)