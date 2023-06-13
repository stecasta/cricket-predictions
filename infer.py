import pandas as pd
from joblib import load
import numpy as np
import argparse


def infer(args):
    df = pd.read_csv(args.data_path)

    if args.test:
        df = df[df["team"] == "Ireland"][:5]

    X = np.sqrt(
        np.array(
            df[
                [
                    "remaining_overs",
                    "remaining_wickets",
                    "innings",
                    "runs_cumul",
                    "runs_needed_to_par",
                    "last_5_overs_mean_runs",
                ]
            ]
        )
    )

    model = load(args.model_path)
    out = model.predict(X)

    df["predicted_runs"] = out
    print(
        df[
            [
                "matchid",
                "team",
                "innings",
                "remaining_overs",
                "remaining_wickets",
                "runs",
                "predicted_runs",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        help="Path to the data file.",
        default="data/intermediate_data.csv",
    )
    parser.add_argument(
        "--model_path",
        help="Path to the smodel file.",
        default="models/cricket_model.pkl",
    )
    parser.add_argument(
        "--test",
        help="Test the script with the first 5 Irelands overs",
        action="store_true",
    )
    args = parser.parse_args()
    infer(args)
