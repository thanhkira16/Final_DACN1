from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import random
from openai import OpenAI
import os
import traceback

app = Flask(__name__)


# Load models
model_paths = {
    'KNN': 'model/knn_model.pkl',
    'RandomForest': 'model/rf_model.pkl',
    'XGBoost': 'model/xgb_model.pkl',
    'HistGradientBoosting': 'model/hist_gb_model.pkl'
}

models = {name: pickle.load(open(path, 'rb')) for name, path in model_paths.items()}

# Manually defined features
feature_names = [
    "CLIENT_GENDER", "AGE", "TENURE", "TOTAL_CREDIT_CARD", "TOTAL_DEBIT_CARD",
    "TOTAL_TERM_DEPOSIT", "AVG_TERM_DEPOSIT_BALANCE", "MAX_TERM_DEPOSIT_BALANCE",
    "TOTAL_LOANS", "AVG_LOAN_BALANCE", "TOTAL_ACTIVITIES", "TOTAL_TRANSACTIONS",
    "TOTAL_TYPE_TRANSACTIONS", "AVG_TRANSACTIONS_NO_MONTH", "AVG_TRANSACTIONS_AMOUNT"
]

# Global variable to store processed data
processed_data = None


def load_processed_data():
    """Load and process data from CSV file"""
    global processed_data
    try:
        # Load CSV file
        df = pd.read_csv('data/data_processed.csv')

        # Filter only required features + CHURN column
        required_columns = feature_names + ['CHURN']

        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in CSV: {missing_columns}")

        # Filter available columns
        available_columns = [col for col in required_columns if col in df.columns]
        df_filtered = df[available_columns].copy()

        # Fill missing feature columns with default values if any
        for feature in feature_names:
            if feature not in df_filtered.columns:
                if feature == "CLIENT_GENDER":
                    df_filtered[feature] = 0
                elif feature == "AGE":
                    df_filtered[feature] = 30
                elif feature == "TENURE":
                    df_filtered[feature] = 12
                else:
                    df_filtered[feature] = 0.0

        # Separate churn and no-churn data
        churn_data = df_filtered[df_filtered['CHURN'] == 1].copy()
        no_churn_data = df_filtered[df_filtered['CHURN'] == 0].copy()

        # Remove CHURN column from feature data
        churn_features = churn_data.drop('CHURN', axis=1, errors='ignore')
        no_churn_features = no_churn_data.drop('CHURN', axis=1, errors='ignore')

        processed_data = {
            'CHURN': churn_features.to_dict('records'),
            'NO_CHURN': no_churn_features.to_dict('records'),
            'total_churn': len(churn_features),
            'total_no_churn': len(no_churn_features)
        }

        print(f"Data loaded successfully:")
        print(f"- CHURN users: {processed_data['total_churn']}")
        print(f"- NO_CHURN users: {processed_data['total_no_churn']}")
        print(f"- Available features: {len(feature_names)}")

        return True

    except FileNotFoundError:
        print("Error: data_processed.csv file not found!")
        return False
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        traceback.print_exc()
        return False


# Load data on startup
if not load_processed_data():
    print("Failed to load CSV data. Using fallback test data.")
    # Fallback test dataset
    processed_data = {
        'CHURN': [
            {
                "CLIENT_GENDER": 1, "AGE": 20, "TENURE": 12,
                "TOTAL_CREDIT_CARD": 12, "TOTAL_DEBIT_CARD": 12,
                "TOTAL_TERM_DEPOSIT": 12, "AVG_TERM_DEPOSIT_BALANCE": 0,
                "MAX_TERM_DEPOSIT_BALANCE": 0, "TOTAL_LOANS": 0,
                "AVG_LOAN_BALANCE": 0, "TOTAL_ACTIVITIES": 54,
                "TOTAL_TRANSACTIONS": 0, "TOTAL_TYPE_TRANSACTIONS": 0,
                "AVG_TRANSACTIONS_NO_MONTH": 0, "AVG_TRANSACTIONS_AMOUNT": 0
            }
        ],
        'NO_CHURN': [
            {
                "CLIENT_GENDER": 1, "AGE": 24, "TENURE": 8,
                "TOTAL_CREDIT_CARD": 9, "TOTAL_DEBIT_CARD": 9,
                "TOTAL_TERM_DEPOSIT": 9, "AVG_TERM_DEPOSIT_BALANCE": 0,
                "MAX_TERM_DEPOSIT_BALANCE": 0, "TOTAL_LOANS": 0,
                "AVG_LOAN_BALANCE": 0, "TOTAL_ACTIVITIES": 67,
                "TOTAL_TRANSACTIONS": 13, "TOTAL_TYPE_TRANSACTIONS": 2,
                "AVG_TRANSACTIONS_NO_MONTH": 1.083333333, "AVG_TRANSACTIONS_AMOUNT": 1633153.846
            }
        ],
        'total_churn': 1,
        'total_no_churn': 1
    }

# API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or "your-api-key-here"

# Initialize OpenAI client for churn prediction
try:
    if OPENAI_API_KEY and OPENAI_API_KEY != "your-api-key-here":
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai_client = None
        print("Warning: OPENAI_API_KEY not properly set. OpenAI churn prediction will be disabled.")
except Exception as e:
    openai_client = None
    print(f"Error initializing OpenAI client: {e}")


def random_input():
    """Generate random input data"""
    random_data = {}
    for feature in feature_names:
        if feature == "CLIENT_GENDER":
            random_data[feature] = random.choice([0, 1])
        elif feature == "AGE":
            random_data[feature] = random.randint(18, 80)
        elif feature == "TENURE":
            random_data[feature] = random.randint(1, 120)
        elif "TOTAL_" in feature:
            random_data[feature] = random.randint(0, 50)
        elif "AVG_" in feature:
            random_data[feature] = round(random.uniform(0, 100000), 2)
        elif "MAX_" in feature:
            random_data[feature] = round(random.uniform(0, 500000), 2)
        else:
            random_data[feature] = round(random.uniform(0, 100), 2)

    return random_data


def get_test_data(dataset_type):
    """Get random test data from CHURN or NO_CHURN dataset loaded from CSV"""
    try:
        if dataset_type.upper() not in processed_data:
            return None, "Invalid dataset type"

        dataset = processed_data[dataset_type.upper()]

        if not dataset:
            return None, f"No {dataset_type} data available"

        selected_user = random.choice(dataset)

        # Ensure all required features are present
        filtered_data = {}
        for feature in feature_names:
            if feature in selected_user:
                filtered_data[feature] = selected_user[feature]
            else:
                # Set default value if feature not found
                if feature == "CLIENT_GENDER":
                    filtered_data[feature] = random.choice([0, 1])
                elif feature == "AGE":
                    filtered_data[feature] = random.randint(18, 80)
                elif feature == "TENURE":
                    filtered_data[feature] = random.randint(1, 120)
                else:
                    filtered_data[feature] = 0.0

        # Generate a fake customer number for display
        customer_number = f"CUST_{random.randint(100000, 999999)}"

        # Get dataset info for message
        total_count = processed_data.get(f'total_{dataset_type.lower()}', len(dataset))

        return filtered_data, f"{customer_number} (from {total_count} {dataset_type} users)"

    except Exception as e:
        print(f"Error in get_test_data: {e}")
        traceback.print_exc()
        return None, f"Error: {str(e)}"


def predict_churn_with_openai(input_data):
    """Predict churn using OpenAI API - returns 0 or 1"""
    if openai_client is None:
        return None, "OpenAI API chưa được cấu hình. Vui lòng thiết lập biến môi trường OPENAI_API_KEY."

    prompt = (
        "Bạn là một chuyên gia phân tích dữ liệu ngân hàng. "
        "Dựa vào dữ liệu khách hàng sau đây, hãy dự đoán khách hàng này có khả năng rời bỏ dịch vụ (churn) hay không. "
        "Chỉ trả lời bằng số 0 (không churn) hoặc 1 (churn), không giải thích thêm.\n\n"
        f"Dữ liệu khách hàng:\n{input_data}"
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )

        result = response.choices[0].message.content.strip()
        # Extract only the number (0 or 1)
        if '1' in result:
            return 1, None
        elif '0' in result:
            return 0, None
        else:
            return None, f"OpenAI trả về kết quả không hợp lệ: {result}"

    except Exception as e:
        return None, f"Lỗi khi gọi OpenAI API: {str(e)}"

def analyze_with_openai(input_data):
    """Phân tích dữ liệu người dùng bằng OpenAI GPT API"""
    if openai_client is None:
        return "Phân tích bằng OpenAI đang bị tắt. Vui lòng thiết lập biến môi trường OPENAI_API_KEY."

    prompt = (
        "Phân tích dữ liệu đầu vào sau đây của một người dùng ứng dụng ngân hàng số. "
        "Đưa ra đánh giá bằng tiếng Việt về trạng thái hoạt động của người dùng này "
        "(ví dụ: đang hoạt động tích cực, hoạt động bình thường, ít hoạt động, có nguy cơ rời bỏ), "
        "kèm theo giải thích ngắn gọn về các yếu tố quan trọng và đề xuất hành động (nếu có). "
        "Giới hạn phản hồi dưới 250 từ.\n\n"
        f"Dữ liệu khách hàng:\n{str(input_data)}"
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Bạn là một chuyên gia phân tích hành vi khách hàng."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Lỗi khi phân tích bằng OpenAI API: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    analysis = None
    input_data = {key: '' for key in feature_names}
    selected_model = 'RandomForest'
    customer_info = None

    if request.method == "POST":
        try:
            if 'random' in request.form:
                input_data = random_input()
            else:
                # Get form data
                input_data = {}
                for key in feature_names:
                    try:
                        value = request.form.get(key, '')
                        if value == '':
                            input_data[key] = 0.0
                        else:
                            input_data[key] = float(value)
                    except ValueError:
                        input_data[key] = 0.0

                selected_model = request.form.get("model", "RandomForest")

                # Make prediction with selected model
                input_df = pd.DataFrame([input_data])
                model = models[selected_model]
                model_prediction = int(model.predict(input_df)[0])

                # Get OpenAI churn prediction
                openai_prediction, openai_error = predict_churn_with_openai(input_data)

                # Determine final prediction (prioritize OpenAI if available)
                if openai_prediction is not None:
                    prediction = openai_prediction  # Use OpenAI result as final prediction
                    if openai_prediction != model_prediction:
                        print(
                            f"Prediction difference - Model: {model_prediction}, OpenAI: {openai_prediction}, Using OpenAI")
                else:
                    prediction = model_prediction  # Use model result if OpenAI fails
                    if openai_error:
                        print(f"OpenAI prediction failed: {openai_error}")

                # Get  analysis
                analysis = analyze_with_openai(input_data)

        except Exception as e:
            print(f"Error in main route: {e}")
            traceback.print_exc()
            customer_info = f"Error: {str(e)}"

    return render_template("index.html",
                           feature_names=feature_names,
                           input_data=input_data,
                           prediction=prediction,
                           analysis=analysis,
                           selected_model=selected_model,
                           customer_info=customer_info,
                           data_stats=processed_data)


@app.route("/load_test_data", methods=["POST"])
def load_test_data():
    """AJAX endpoint to load test data"""
    try:
        data_json = request.get_json()

        if not data_json:
            return jsonify({
                "success": False,
                "message": "No JSON data received"
            }), 400

        dataset_type = data_json.get("dataset_type")

        if not dataset_type:
            return jsonify({
                "success": False,
                "message": "dataset_type is required"
            }), 400

        test_result = get_test_data(dataset_type)

        if test_result[0] is not None:  # test_result is (data, customer_info)
            input_data, customer_info = test_result
            return jsonify({
                "success": True,
                "data": input_data,
                "customer_info": customer_info,
                "message": f"Loaded {dataset_type} user: {customer_info}"
            })
        else:
            return jsonify({
                "success": False,
                "message": test_result[1]  # Error message
            }), 500

    except Exception as e:
        print(f"Error in load_test_data: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500


@app.route("/predict_ajax", methods=["POST"])
def predict_ajax():
    """AJAX endpoint for prediction"""
    try:
        data_json = request.get_json()

        if not data_json:
            return jsonify({"error": "No JSON data received"}), 400

        model_name = data_json.pop("model", "RandomForest")

        # Ensure all required features are present
        input_data = {}
        for feature in feature_names:
            if feature in data_json:
                input_data[feature] = data_json[feature]
            else:
                input_data[feature] = 0.0

        # Make prediction with selected model
        input_df = pd.DataFrame([input_data])
        model = models[model_name]
        model_prediction = int(model.predict(input_df)[0])

        # Get OpenAI churn prediction
        openai_prediction, openai_error = predict_churn_with_openai(input_data)

        # Determine final prediction (prioritize OpenAI if available)
        if openai_prediction is not None:
            prediction = openai_prediction  # Use OpenAI result as final prediction
        else:
            prediction = model_prediction  # Use model result if OpenAI fails

        # Get Gemini analysis
        analysis = analyze_with_openai(input_data)

        return jsonify({
            "prediction": prediction,
            "analysis": analysis
        })

    except Exception as e:
        print(f"Error in predict_ajax: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Server error: {str(e)}"
        }), 500


@app.route("/reload_data", methods=["POST"])
def reload_data():
    """AJAX endpoint to reload data from CSV"""
    try:
        success = load_processed_data()
        if success:
            return jsonify({
                "success": True,
                "message": f"Data reloaded successfully. CHURN: {processed_data['total_churn']}, NO_CHURN: {processed_data['total_no_churn']}",
                "stats": {
                    "total_churn": processed_data['total_churn'],
                    "total_no_churn": processed_data['total_no_churn']
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to reload data from CSV"
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error reloading data: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)