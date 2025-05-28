from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import random
from openai import OpenAI
import os

app = Flask(__name__)

# Load models
model_paths = {
    'KNN': 'model/knn_model.pkl',
    'RandomForest': 'model/rf_model.pkl',
    'XGBoost': 'model/xgb_model.pkl',
    'HistGradientBoosting': 'model/hist_gb_model.pkl'
}
models = {name: pickle.load(open(path, 'rb')) for name, path in model_paths.items()}

# Load data
data = pd.read_csv('data/data_processed.csv')
feature_names = data.columns.drop('CHURN').tolist()

# OpenAI ChatGPT API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or "your-api-key-here"  # Replace with your actual API key

# Initialize OpenAI client with error handling
try:
    if OPENAI_API_KEY and OPENAI_API_KEY != "your-api-key-here":
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        client = None
        print("Warning: OPENAI_API_KEY not properly set. ChatGPT analysis will be disabled.")
except Exception as e:
    client = None
    print(f"Error initializing OpenAI client: {e}")


def random_input():
    row = data.sample(1).iloc[0]
    return {col: float(row[col]) for col in feature_names}


def analyze_with_chatgpt(input_data):
    if client is None:
        return "Phân tích bằng ChatGPT đang bị tắt. Vui lòng thiết lập biến môi trường OPENAI_API_KEY."

    prompt = (
        "Phân tích dữ liệu đầu vào sau đây của một người dùng ứng dụng ngân hàng số. "
        "Đưa ra đánh giá bằng tiếng Việt về trạng thái hoạt động của người dùng này (ví dụ: đang hoạt động, không hoạt động, có nguy cơ rời bỏ), "
        "kèm theo giải thích ngắn gọn dưới 200 từ. "
        "Dữ liệu: " + str(input_data)
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Lỗi khi phân tích bằng ChatGPT API: {str(e)}"



@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    analysis = None
    input_data = {key: '' for key in feature_names}
    selected_model = 'RandomForest'

    if request.method == "POST":
        if 'random' in request.form:
            input_data = random_input()
        else:
            try:
                input_data = {key: float(request.form[key]) for key in feature_names}
            except:
                input_data = {key: request.form[key] for key in feature_names}

            selected_model = request.form.get("model", "RandomForest")
            input_df = pd.DataFrame([input_data])
            model = models[selected_model]
            prediction = int(model.predict(input_df)[0])

            # Call ChatGPT API for analysis
            analysis = analyze_with_chatgpt(input_data)

    return render_template("index.html",
                           feature_names=feature_names,
                           input_data=input_data,
                           prediction=prediction,
                           analysis=analysis,
                           selected_model=selected_model)


@app.route("/predict_ajax", methods=["POST"])
def predict_ajax():
    data_json = request.get_json()
    model_name = data_json.pop("model")
    input_df = pd.DataFrame([data_json])
    model = models[model_name]
    prediction = int(model.predict(input_df)[0])
    analysis = analyze_with_chatgpt(data_json)
    return jsonify({"prediction": prediction, "analysis": analysis})


if __name__ == "__main__":
    app.run(debug=True)