import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_groundwater_model():
    data_path = 'Groundwater.csv'
    if not os.path.exists(data_path):
        print("Data file not found!")
        return

    df = pd.read_csv(data_path)

    # --- Feature Engineering ---
    # Create a 'Usage Intensity' feature
    df['Usage_Intensity'] = df['Total_Usage'] / (df['Net annual groundwater availability'] + 1e-5)
    
    # Create 'Recharge Efficiency'
    df['Recharge_Efficiency'] = (df['Recharge from rainfallMonsoon season'] + df['Recharge from rainfallNon-monsoon season']) / (df['Total_Rainfall'] + 1e-5)

    # Prepare features and target
    # Dropping 'States' as it's a unique identifier, and 'Situation' which is target
    X = df.drop(columns=['States', 'Situation'])
    y = df['Situation'].str.strip() # Clean whitespace

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    predictions = model.predict(X_test_scaled)
    print("Model Evaluation:")
    print(classification_report(y_test, predictions))

    # Save Model, Scaler, and Feature Names (important for dashboard)
    joblib.dump(model, 'groundwater_model.pkl')
    joblib.dump(scaler, 'groundwater_scaler.pkl')
    joblib.dump(list(X.columns), 'feature_names.pkl')
    
    print("Model, Scaler, and Feature names saved successfully!")

if __name__ == "__main__":
    train_groundwater_model()
