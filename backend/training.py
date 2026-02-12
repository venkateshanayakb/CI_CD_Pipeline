import pandas as pd
import pickle
from pathlib import Path
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "inpatientCharges.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.pkl"
METRICS_PATH = Path(__file__).resolve().parent / "models" / "metrics.json"

# --------------------------------------------------
# Load Data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

money_cols = [
    "Average Covered Charges",
    "Average Total Payments",
    "Average Medicare Payments"
]

for col in money_cols:
    df[col] = df[col].replace(r'[$,]', '', regex=True).astype(float)

features = ["Total Discharges", "Provider State", "DRG Definition"]
target = "Average Medicare Payments"

X = df[features]
y = np.log(df[target])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

numeric_features = ["Total Discharges"]
categorical_features = ["Provider State", "DRG Definition"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# --------------------------------------------------
# Models
# --------------------------------------------------

def evaluate_model(name, pipeline, X_test, y_test):
    log_preds = pipeline.predict(X_test)

    # Convert back to original scale
    preds = np.exp(log_preds)
    y_test_original = np.exp(y_test)

    mae = mean_absolute_error(y_test_original, preds)
    rmse = np.sqrt(mean_squared_error(y_test_original, preds))
    r2 = r2_score(y_test_original, preds)
    mape = np.mean(np.abs((y_test_original - preds) / y_test_original)) * 100


    print(f"\n{name} Performance:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")

    return {
        "pipeline": pipeline,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


# Create pipelines
linear_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# Train
linear_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# Evaluate
linear_results = evaluate_model(
    "Linear Regression",
    linear_pipeline,
    X_test,
    y_test
)

rf_results = evaluate_model(
    "Random Forest",
    rf_pipeline,
    X_test,
    y_test
)

# --------------------------------------------------
# Compare Models
# --------------------------------------------------

print("\nModel Comparison (Lower MAE is better):")

if linear_results["MAE"] < rf_results["MAE"]:
    best_model_name = "LinearRegression"
    best_pipeline = linear_results["pipeline"]
    print("Linear Regression performs better based on MAE.")
else:
    best_model_name = "RandomForest"
    best_pipeline = rf_results["pipeline"]
    print("Random Forest performs better based on MAE.")

# --------------------------------------------------
# Save Best Model
# --------------------------------------------------

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_pipeline, f)

metrics_to_save = {
    "selected_model": best_model_name,
    "metrics": {
        "LinearRegression": {
            "MAE": linear_results["MAE"],
            "RMSE": linear_results["RMSE"],
            "R2": linear_results["R2"],
            "MAPE": linear_results["MAPE"]
        },
        "RandomForest": {
            "MAE": rf_results["MAE"],
            "RMSE": rf_results["RMSE"],
            "R2": rf_results["R2"],
            "MAPE": rf_results["MAPE"]
        }
    }
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics_to_save, f, indent=4)

print("\nBest model selected:", best_model_name)
print("Model and metrics saved successfully.")

# ------------------------------------------
# Error Analysis for Linear Regression
# ------------------------------------------

preds = linear_pipeline.predict(X_test)

residuals = y_test - preds
abs_errors = np.abs(residuals)

error_df = X_test.copy()
error_df["Actual"] = y_test
error_df["Predicted"] = preds
error_df["Residual"] = residuals
error_df["Absolute Error"] = abs_errors
error_df["Percentage Error"] = (abs_errors / y_test) * 100

#Basic Error Distribution
print("\nError Distribution Summary:")
print(error_df["Absolute Error"].describe())

print("\nTop 10 Worst Predictions:")
print(error_df.sort_values("Absolute Error", ascending=False).head(10))

#Error by State
state_error = error_df.groupby("Provider State")["Absolute Error"].mean().sort_values(ascending=False)

print("\nAverage Absolute Error by State:")
print(state_error.head(10))

#Error vs Payment Size
error_df["Payment Bucket"] = pd.qcut(error_df["Actual"], 5)

bucket_error = error_df.groupby("Payment Bucket")["Absolute Error"].mean()

print("\nAverage Error by Payment Range:")
print(bucket_error)

#Quick Visualization
import matplotlib.pyplot as plt

plt.scatter(error_df["Actual"], error_df["Absolute Error"], alpha=0.3)
plt.xlabel("Actual Medicare Payment")
plt.ylabel("Absolute Error")
plt.title("Error vs Payment Size")
plt.show()
