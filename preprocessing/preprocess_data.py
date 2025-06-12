import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

RAW_DATA_PATH = "../data/raw/flood_data.csv"
PROCESSED_DIR = "../data/processed"

def preprocess_data():
    df = pd.read_csv(RAW_DATA_PATH)
    
    # No missing values, but this is a safety net
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Features and target
    X = df.drop(columns=['FloodProbability'])
    y = df['FloodProbability']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ensure directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Save the scaler and data
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, 'feature_scaler.pkl'))
    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train.values)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test.values)

if __name__ == "__main__":
    preprocess_data()
    print("Data preprocessing complete. Processed data saved to", PROCESSED_DIR)