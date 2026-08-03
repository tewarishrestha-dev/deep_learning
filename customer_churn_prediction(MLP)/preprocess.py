import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():

    path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "Telco-Customer-Churn.csv"
    )

    df = pd.read_csv(path)

    # Remove unnecessary column
    df.drop(columns="customerID", inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # Binary Encoding
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    binary_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn"
    ]
    
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # One-Hot Encoding
    df = pd.get_dummies(
        df,
        columns=[
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaymentMethod"
        ],
        dtype=int
    )

    # Features & Labels
    X = df.drop(columns="Churn").values
    y = df["Churn"].values.reshape(-1, 1)

    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    print("Training Samples :", X_train.shape)
    print("Testing Samples  :", X_test.shape)
    print("Train Labels     :", y_train.shape)
    print("Test Labels      :", y_test.shape)