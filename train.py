from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from joblib import dump
import argparse
import pandas as pd
from joblib import dump
import numpy as np


def train(args):
    n_test_samples = 5000  # Leave out test set for reproducibility
    df = pd.read_csv(args.data_path)

    X = df[:-n_test_samples][
        [
            "remaining_overs",
            "remaining_wickets",
            "last_5_overs_mean_runs",
            "runs_cumul",
            "runs_needed_to_par",
            "innings",
        ]
    ]
    y = df[:-n_test_samples]["runs"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=6,
        eta=0.01,
        subsample=0.7,
        colsample_bytree=0.8,
    )
    print("Training model...")
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_val)
    r2 = r2_score(y_test, y_pred_val)
    print(f"RMSE train: {np.sqrt(mse_train)}")
    print(f"RMSE test: {np.sqrt(mse_test)}")
    print(f"R2 score test: {r2}")

    if dump(model, args.model_save_path):
        print("Model saved to:", args.model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        help="Path to the data file.",
        default="data/intermediate_data.csv",
    )
    parser.add_argument(
        "--model-save-path",
        help="Output path for the model",
        default="models/cricket_model.pkl",
    )
    args = parser.parse_args()
    train(args)
