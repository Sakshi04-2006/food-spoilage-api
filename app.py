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
        data = request.get_json()

        food_type = int(data["food_type"])  # 0 or 1
        cooking_temp = float(data["cooking_temp"])
        storage_temp = float(data["storage_temp"])
        storage_time = float(data["storage_time"])

        input_df = pd.DataFrame(
            [[food_type, cooking_temp, storage_temp, storage_time]],
            columns=["Food_Type","Cooking_Temp","Storage_Temp","Storage_Time"]
        )

        prediction = model.predict(input_df)

        result = prediction[0]

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500