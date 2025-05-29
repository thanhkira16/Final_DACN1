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

# Test dataset
TEST_DATASET = {
    "CHURN": [
        {
            "CUSTOMER_NUMBER": 452440,
            "CLIENT_GENDER": 1,
            "CLIENT_AGE": 20,
            "STAFF_VIB": 0,
            "ACCOUNT_AGE_DAYS": 355,
            "SMS": 1,
            "VERIFY_METHOD": 1,
            "EB_REGISTER_CHANNEL": 0,
            "TOTAL_CREDIT_CARD": 12,
            "TOTAL_DEBIT_CARD": 12,
            "TOTAL_CURRENT_ACCOUNTS": 12,
            "BALANCE": 133671.0242,
            "MAX_CURRENT_ACCOUNT_BALANCE": 188607.14,
            "MIN_CURRENT_ACCOUNT_BALANCE": 78645.16,
            "TOTAL_TERM_DEPOSIT": 12,
            "AVG_TERM_DEPOSIT_BALANCE": 0,
            "MAX_TERM_DEPOSIT_BALANCE": 0,
            "MIN_TERM_DEPOSIT_BALANCE": 0,
            "TOTAL_LOANS": 0,
            "AVG_LOAN_BALANCE": 0,
            "MAX_LOAN_BALANCE": 0,
            "MIN_LOAN_BALANCE": 0,
            "TOTAL_ACTIVITIES": 54,
            "TOTAL_TRANSACTIONS": 0,
            "TOTAL_TRANSACTIONS_AMOUNT": 0,
            "MAX_TRANSACTIONS_AMOUNT": 0,
            "MIN_TRANSACTIONS_AMOUNT": 0,
            "TOTAL_TYPE_TRANSACTIONS": 0,
            "AVG_TRANSACTIONS_NO_MONTH": 0,
            "AVG_TRANSACTION_AMOUNT": 0,
            "LAST_ACTIVITY_DAYS": 30
        }
    ],
    "NO_CHURN": [
        {
            "CUSTOMER_NUMBER": 639362,
            "CLIENT_GENDER": 1,
            "CLIENT_AGE": 24,
            "STAFF_VIB": 0,
            "ACCOUNT_AGE_DAYS": 253,
            "SMS": 1,
            "VERIFY_METHOD": 0,
            "EB_REGISTER_CHANNEL": 0,
            "TOTAL_CREDIT_CARD": 9,
            "TOTAL_DEBIT_CARD": 9,
            "TOTAL_CURRENT_ACCOUNTS": 9,
            "BALANCE": 1653584.826,
            "MAX_CURRENT_ACCOUNT_BALANCE": 9371139.87,
            "MIN_CURRENT_ACCOUNT_BALANCE": 32600,
            "TOTAL_TERM_DEPOSIT": 9,
            "AVG_TERM_DEPOSIT_BALANCE": 0,
            "MAX_TERM_DEPOSIT_BALANCE": 0,
            "MIN_TERM_DEPOSIT_BALANCE": 0,
            "TOTAL_LOANS": 0,
            "AVG_LOAN_BALANCE": 0,
            "MAX_LOAN_BALANCE": 0,
            "MIN_LOAN_BALANCE": 0,
            "TOTAL_ACTIVITIES": 67,
            "TOTAL_TRANSACTIONS": 13,
            "TOTAL_TRANSACTIONS_AMOUNT": 21231000,
            "MAX_TRANSACTIONS_AMOUNT": 10000000,
            "MIN_TRANSACTIONS_AMOUNT": 20000,
            "TOTAL_TYPE_TRANSACTIONS": 2,
            "AVG_TRANSACTIONS_NO_MONTH": 1.083333333,
            "AVG_TRANSACTION_AMOUNT": 1633153.846,
            "LAST_ACTIVITY_DAYS": 5
        }
    ]
}

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
    """Generate random input from existing data"""
    row = data.sample(1).iloc[0]
    return {col: float(row[col]) for col in feature_names}


def get_test_data(dataset_type):
    """Get random test data from CHURN or NO_CHURN dataset"""
    if dataset_type.upper() not in TEST_DATASET:
        return None

    dataset = TEST_DATASET[dataset_type.upper()]
    selected_user = random.choice(dataset)

    # Filter only the features that exist in feature_names
    filtered_data = {}
    for feature in feature_names:
        if feature in selected_user:
            filtered_data[feature] = selected_user[feature]
        else:
            # Set default value if feature not found
            filtered_data[feature] = 0.0

    return filtered_data, selected_user["CUSTOMER_NUMBER"]


def analyze_with_chatgpt(input_data):
    """Analyze user data with ChatGPT"""
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
    customer_info = None

    if request.method == "POST":
        if 'random' in request.form:
            input_data = random_input()
        elif 'load_test_data' in request.form:
            dataset_type = request.form['dataset_type']
            test_result = get_test_data(dataset_type)
            if test_result:
                input_data, customer_number = test_result
                customer_info = f"Loaded {dataset_type} user (Customer: {customer_number})"
            else:
                customer_info = "Error loading test data"
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
                           selected_model=selected_model,
                           customer_info=customer_info)


@app.route("/load_test_data", methods=["POST"])
def load_test_data():
    """AJAX endpoint to load test data"""
    data_json = request.get_json()
    dataset_type = data_json.get("dataset_type")

    test_result = get_test_data(dataset_type)
    if test_result:
        input_data, customer_number = test_result
        return jsonify({
            "success": True,
            "data": input_data,
            "customer_number": customer_number,
            "message": f"Loaded {dataset_type} user (Customer: {customer_number})"
        })
    else:
        return jsonify({
            "success": False,
            "message": "Error loading test data"
        })


@app.route("/predict_ajax", methods=["POST"])
def predict_ajax():
    """AJAX endpoint for prediction"""
    data_json = request.get_json()
    model_name = data_json.pop("model")
    input_df = pd.DataFrame([data_json])
    model = models[model_name]
    prediction = int(model.predict(input_df)[0])
    analysis = analyze_with_chatgpt(data_json)
    return jsonify({"prediction": prediction, "analysis": analysis})


if __name__ == "__main__":
    app.run(debug=True)