import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define column mappings and validations
COLUMN_MAPPINGS = {
    'Churn': {'Yes': 1, 'No': 0},
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 0},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'PaymentMethod': {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }
}

def validate_dataframe(df):
    """Validate the dataframe structure and data types."""
    logger.info("Validating dataframe...")
    
    # Check required columns
    required_columns = list(COLUMN_MAPPINGS.keys()) + ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate numeric columns
    numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def encode_categorical_columns(df):
    """Encode categorical columns using predefined mappings."""
    logger.info("Encoding categorical columns...")
    df_encoded = df.copy()
    
    for column, mapping in COLUMN_MAPPINGS.items():
        if column in df_encoded.columns:
            logger.debug(f"Encoding column: {column}")
            df_encoded[column] = df_encoded[column].map(mapping)
            # Fill any NaN values with the most common value
            if df_encoded[column].isna().any():
                most_common = df_encoded[column].mode()[0]
                df_encoded[column] = df_encoded[column].fillna(most_common)
    
    return df_encoded

def load_data(file_path=None):
    """Load the dataset from the specified file path."""
    logger.info("Starting data loading process...")
    if file_path is None:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    try:
        logger.info(f"Attempting to load data from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Validate and clean the data
        df = validate_dataframe(df)
        return df
    except FileNotFoundError:
        logger.error(f"Dataset not found at {file_path}")
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please ensure the file exists.")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise Exception(f"Error loading dataset: {str(e)}")

def preprocess_data(df):
    """Preprocess the data by handling missing values and encoding categorical variables."""
    logger.info("Starting data preprocessing...")
    
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Store customerID for later use
    customer_ids = df_processed['customerID']
    
    # Handle missing values in numeric columns
    logger.info("Handling missing values...")
    numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Encode categorical columns
    df_processed = encode_categorical_columns(df_processed)
    
    # Convert SeniorCitizen to int if it's not already
    df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(int)
    
    # Add customerID back
    df_processed['customerID'] = customer_ids
    
    logger.info("Data preprocessing completed successfully")
    return df_processed

def feature_engineering(df):
    """Create new features from existing data."""
    logger.info("Starting feature engineering...")
    
    df_engineered = df.copy()
    
    # Store customerID for later use
    customer_ids = df_engineered['customerID']
    
    # Create tenure groups
    logger.info("Creating tenure groups...")
    if len(df_engineered) > 1:
        df_engineered['tenure_group'] = pd.qcut(df_engineered['tenure'], q=4, labels=[0, 1, 2, 3])
    else:
        # For single-row input, assign default group (e.g., 0)
        df_engineered['tenure_group'] = 0
    
    # Calculate monthly charges per service
    logger.info("Calculating charges per service...")
    service_columns = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                      'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                      'StreamingTV', 'StreamingMovies']
    
    df_engineered['charges_per_service'] = df_engineered['MonthlyCharges'] / (
        df_engineered[service_columns].sum(axis=1)
    )
    
    # Fill any infinite values with 0
    df_engineered['charges_per_service'] = (
        df_engineered['charges_per_service'].replace([np.inf, -np.inf], 0).fillna(0)
    )
    
    # Add customerID back
    df_engineered['customerID'] = customer_ids
    
    logger.info("Feature engineering completed successfully")
    return df_engineered

def prepare_data_for_training(df, target_column='Churn', test_size=0.2, random_state=42):
    """Prepare the data for model training."""
    logger.info("Preparing data for training...")
    
    # Store customerID for later use
    customer_ids = df['customerID']
    
    # Separate features and target
    X = df.drop(columns=[target_column, 'customerID'])
    y = df[target_column]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Split the data
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Data preparation completed successfully")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    # Example usage
    logger.info("Starting data preprocessing pipeline...")
    df = load_data()
    df_processed = preprocess_data(df)
    df_engineered = feature_engineering(df_processed)
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(df_engineered)
    
    logger.info("Data preprocessing completed successfully!")
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Testing set shape: {X_test.shape}") 