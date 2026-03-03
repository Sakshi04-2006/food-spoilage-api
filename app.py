from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("food_model.pkl")

@app.route("/")
def home():
    return "Food Spoilage API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        if data is None:
            return jsonify({"error": "No JSON received"}), 400

        food_type = int(data.get("food_type", 0))
        cooking_temp = float(data.get("cooking_temp", 0))
        storage_temp = float(data.get("storage_temp", 0))
        storage_time = float(data.get("storage_time", 0))

        input_df = pd.DataFrame(
            [[food_type, cooking_temp, storage_temp, storage_time]],
            columns=["Food_Type","Cooking_Temp","Storage_Temp","Storage_Time"]
        )

        prediction = model.predict(input_df)

        return jsonify({"result": str(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)