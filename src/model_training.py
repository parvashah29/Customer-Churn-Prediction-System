import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import joblib
import os
import logging
from data_preprocessing import load_data, preprocess_data, feature_engineering, prepare_data_for_training

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost model."""
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    logger.info("Fitting XGBoost model...")
    model.fit(X_train, y_train)
    
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    logger.info("XGBoost model training completed")
    return model, metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest model."""
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    logger.info("Fitting Random Forest model...")
    model.fit(X_train, y_train)
    
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    logger.info("Random Forest model training completed")
    return model, metrics

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Logistic Regression model."""
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    
    logger.info("Fitting Logistic Regression model...")
    model.fit(X_train, y_train)
    
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    logger.info("Logistic Regression model training completed")
    return model, metrics

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate model performance metrics."""
    logger.info("Calculating model metrics...")
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }
    logger.debug(f"Calculated metrics: {metrics}")
    return metrics

def save_model(model, model_name, scaler=None):
    """Save the trained model and scaler to disk."""
    logger.info(f"Saving {model_name}...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = f'models/{model_name}.joblib'
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save scaler if provided
    if scaler is not None:
        scaler_path = f'models/{model_name}_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

def load_model(model_name, scaler=False):
    """Load a trained model from disk."""
    logger.info(f"Loading {model_name}...")
    try:
        model_path = f'models/{model_name}.joblib'
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        if scaler:
            scaler_path = f'models/{model_name}_scaler.joblib'
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
            return model, scaler
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise

def compare_models(models_metrics):
    """Compare the performance of different models."""
    logger.info("Comparing model performances...")
    comparison = pd.DataFrame(models_metrics).T
    logger.info("\nModel Comparison:\n" + str(comparison))
    return comparison

if __name__ == "__main__":
    logger.info("Starting model training pipeline...")
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = load_data()
    df_processed = preprocess_data(df)
    df_engineered = feature_engineering(df_processed)
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(df_engineered)
    
    # Train models
    logger.info("Training models...")
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    # Compare models
    models_metrics = {
        'XGBoost': xgb_metrics,
        'Random Forest': rf_metrics,
        'Logistic Regression': lr_metrics
    }
    comparison = compare_models(models_metrics)
    
    # Save best model (XGBoost)
    logger.info("Saving best model (XGBoost)...")
    save_model(xgb_model, 'xgboost_model', scaler)
    
    logger.info("Model training pipeline completed successfully!") 