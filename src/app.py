from flask import Flask, render_template, request, jsonify, session, send_from_directory
import pandas as pd
import numpy as np
import joblib
from data_preprocessing import preprocess_data, feature_engineering
from model_training import load_model
import logging
import plotly.express as px
import json
import plotly

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

def load_data():
    """Load the customer churn dataset."""
    try:
        # Try multiple possible paths for the dataset
        possible_paths = [
            'data/customer_churn.csv',
            'data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
            '../data/customer_churn.csv',
            '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
        ]
        
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                logger.info(f"Successfully loaded data from {path}")
                return df
            except FileNotFoundError:
                continue
        
        # If no file is found, raise an error
        raise FileNotFoundError("Customer churn dataset not found. Please ensure the data file is in the correct location.")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page."""
    # Check if data is available
    df = load_data()
    if df is None:
        return render_template('error.html', 
                             message="Unable to load customer data. Please ensure the data file is in the correct location.",
                             details="The application is looking for the customer churn dataset in the 'data' directory.")
    return render_template('dashboard.html')

@app.route('/predict')
def predict_page():
    """Render the prediction form page."""
    return render_template('predict.html')

@app.route('/api/dashboard-data')
def get_dashboard_data():
    """API endpoint to get dashboard data."""
    try:
        df = load_data()
        if df is None:
            return jsonify({
                'error': 'Failed to load data',
                'message': 'Please ensure the customer churn dataset is in the correct location.'
            }), 500

        df_processed = preprocess_data(df)
        df_engineered = feature_engineering(df_processed)

        # Calculate key metrics
        churn_rate = df_processed['Churn'].mean() * 100
        avg_tenure = df['tenure'].mean()
        avg_charges = df['MonthlyCharges'].mean()

        # Create churn distribution plot
        churn_counts = df['Churn'].value_counts()
        churn_fig = px.pie(values=churn_counts.values, names=churn_counts.index,
                          title='Churn Distribution',
                          color_discrete_map={'Yes': 'red', 'No': 'green'})
        churn_plot = json.dumps(churn_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Service usage analysis
        services = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies']
        
        service_data = []
        for service in services:
            service_churn = df.groupby(service)['Churn'].apply(
                lambda x: (x == 'Yes').mean() * 100
            ).reset_index()
            service_churn.columns = ['Service', 'Churn Rate']
            service_churn['Service Type'] = service
            service_data.append(service_churn)
        
        service_df = pd.concat(service_data)
        service_fig = px.bar(service_df, x='Service Type', y='Churn Rate',
                           color='Service', title='Churn Rate by Service')
        service_plot = json.dumps(service_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Contract analysis
        contract_churn = df.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        contract_churn.columns = ['Contract', 'Churn Rate']
        contract_fig = px.bar(contract_churn, x='Contract', y='Churn Rate',
                            title='Churn Rate by Contract Type')
        contract_plot = json.dumps(contract_fig, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify({
            'metrics': {
                'churn_rate': round(churn_rate, 1),
                'avg_tenure': round(avg_tenure, 1),
                'avg_charges': round(avg_charges, 2)
            },
            'plots': {
                'churn_distribution': churn_plot,
                'service_analysis': service_plot,
                'contract_analysis': contract_plot
            }
        })

    except Exception as e:
        logger.error(f"Error in dashboard data: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'An error occurred while processing the dashboard data.'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions."""
    try:
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'customerID': ['PRED-001'],
            'gender': [data['gender']],
            'SeniorCitizen': [int(data['senior_citizen'])],
            'Partner': [str(data['partner'])],
            'Dependents': [str(data['dependents'])],
            'tenure': [int(data['tenure'])],
            'PhoneService': [str(data['phone_service'])],
            'MultipleLines': [data['multiple_lines']],
            'InternetService': [data['internet_service']],
            'OnlineSecurity': [data['online_security']],
            'OnlineBackup': [data['online_backup']],
            'DeviceProtection': [data['device_protection']],
            'TechSupport': [data['tech_support']],
            'StreamingTV': [data['streaming_tv']],
            'StreamingMovies': [data['streaming_movies']],
            'Contract': [data['contract']],
            'PaperlessBilling': [str(data['paperless_billing'])],
            'PaymentMethod': [data['payment_method']],
            'MonthlyCharges': [float(data['monthly_charges'])],
            'TotalCharges': [float(data['monthly_charges']) * int(data['tenure'])]
        })

        logger.info("Input data created successfully")
        
        # Load model and scaler
        try:
            model, scaler = load_model('xgboost_model', scaler=True)
            logger.info("Model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return jsonify({'error': 'Failed to load model'}), 500
        
        # Preprocess input data
        try:
            input_processed = preprocess_data(input_data)
            input_engineered = feature_engineering(input_processed)
            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return jsonify({'error': 'Failed to preprocess data'}), 500
        
        # Make prediction
        try:
            # Drop customerID before scaling
            features = input_engineered.drop('customerID', axis=1)
            input_scaled = scaler.transform(features)
            churn_probability = model.predict_proba(input_scaled)[0][1]
            logger.info(f"Prediction made successfully: {churn_probability}")
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return jsonify({'error': 'Failed to make prediction'}), 500
        
        # Determine risk level
        risk_level = "High" if churn_probability > 0.7 else "Medium" if churn_probability > 0.3 else "Low"
        
        return jsonify({
            'churn_probability': round(churn_probability * 100, 1),
            'risk_level': risk_level
        })

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True) 