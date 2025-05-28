# 📊 Customer Churn Prediction Web App

## 🚀 Overview

This project is a Flask-based web application that predicts customer churn (whether a customer is likely to leave a service) using various machine learning models. Users can manually input customer data or generate random values, then select a model to perform the prediction.

## 🧠 Machine Learning Models

The application supports the following trained models:

- 🟢 **K-Nearest Neighbors (KNN)**
- 🌲 **Random Forest Classifier**
- 📈 **Histogram-based Gradient Boosting (HistGradientBoosting)**
- ⚡ **XGBoost Classifier**

## 📚 Technologies Used

- **Flask** - Web framework
- **scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **NumPy** - Numerical computing
- **Pickle** - Model serialization

## 🗂 Project Structure

```
Final_DACN/
├── app.py                    # Main Flask application
├── templates/
│   └── index.html           # HTML UI with form and layout
├── static/
│   └── style.css            # Custom CSS (optional)
├── models/
│   ├── rf_model.pkl         # Random Forest model
│   ├── knn_model.pkl        # KNN model
│   ├── hist_gb_model.pkl    # HistGradientBoosting model
│   └── xgb_model.json      # XGBoost model (saved with native API)
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 💻 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

#### 1. Clone the repository

```bash
git clone <repository-url>
cd Final_DACN
```

#### 2. Create and activate a virtual environment (recommended)

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:
```bash
pip install flask scikit-learn xgboost numpy pandas
```

#### 4. Run the Flask application

```bash
python app.py
```

#### 5. Access the application

Visit `http://127.0.0.1:5000` in your browser to access the app.

## 🎯 How to Use

1. **Input Customer Data**: Enter customer characteristics in the form fields
2. **Select Model**: Choose from KNN, Random Forest, HistGradientBoosting, or XGBoost
3. **Make Prediction**: Click "Predict" button to get churn prediction
4. **Random Input**: Use "Random Input" button to automatically generate sample data
5. **View Results**: Prediction results will be displayed below the form

## ✨ Features

- 🖥️ User-friendly web interface
- 📝 Manual data entry or random data generation
- 🤖 Multiple ML model options
- ⚡ Real-time predictions
- 🎨 Styled UI using HTML and CSS
- 📱 Responsive design

