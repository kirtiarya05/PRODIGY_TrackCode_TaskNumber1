# house_price_model_full.py
# End-to-end ML pipeline for house price prediction
# Linear + Ridge Regression with preprocessing and feature engineering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print("\nDataset loaded successfully")
    print("Shape:", df.shape)
    return df


# ------------------------------------------------
# 2. Basic Exploration
# ------------------------------------------------
def explore_data(df):
    print("\nFirst rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head())

    print("\nSummary statistics:")
    print(df.describe())


# ------------------------------------------------
# 3. Feature Engineering + Cleaning
# ------------------------------------------------
def preprocess_data(df):

    # Select important features
    features = [
        "GrLivArea",
        "OverallQual",
        "GarageCars",
        "TotalBsmtSF",
        "FullBath",
        "YearBuilt",
        "TotRmsAbvGrd",
        "SalePrice"
    ]

    df = df[features]

    # Remove missing values
    df = df.dropna()

    # Remove extreme outliers
    df = df[df["GrLivArea"] < 4000]

    # Feature engineering
    df["HouseAge"] = 2024 - df["YearBuilt"]

    print("\nData after preprocessing:", df.shape)

    return df


# ------------------------------------------------
# 4. Data Visualization
# ------------------------------------------------
def visualize_data(df):

    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()


# ------------------------------------------------
# 5. Prepare Train/Test Data
# ------------------------------------------------
def prepare_data(df):

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    return X_train, X_test, y_train, y_test, X.columns


# ------------------------------------------------
# 6. Build Model Pipeline
# ------------------------------------------------
def build_model():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])

    return pipeline


# ------------------------------------------------
# 7. Train Model
# ------------------------------------------------
def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)

    print("\nModel training completed")

    return model


# ------------------------------------------------
# 8. Evaluate Model
# ------------------------------------------------
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance")
    print("----------------------")
    print("MAE :", round(mae,2))
    print("RMSE:", round(rmse,2))
    print("R² Score:", round(r2,3))

    return y_pred


# ------------------------------------------------
# 9. Feature Importance
# ------------------------------------------------
def show_feature_importance(model, feature_names):

    coefficients = model.named_steps["model"].coef_

    print("\nFeature Impact")
    print("----------------------")

    for feature, coef in zip(feature_names, coefficients):
        print(f"{feature:15} : {coef:.2f}")


# ------------------------------------------------
# 10. Plot Predictions
# ------------------------------------------------
def plot_predictions(y_test, y_pred):

    plt.figure(figsize=(6,6))

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")

    plt.show()


# ------------------------------------------------
# 11. Main Program
# ------------------------------------------------
def main():

    filepath = "train.csv"

    df = load_data(filepath)

    explore_data(df)

    df = preprocess_data(df)

    visualize_data(df)

    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    model = build_model()

    model = train_model(model, X_train, y_train)

    y_pred = evaluate_model(model, X_test, y_test)

    show_feature_importance(model, feature_names)

    plot_predictions(y_test, y_pred)


if __name__ == "__main__":
    main()
