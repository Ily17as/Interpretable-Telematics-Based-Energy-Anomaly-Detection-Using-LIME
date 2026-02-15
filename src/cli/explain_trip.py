from __future__ import annotations

"""
Command‑line tool to generate a LIME explanation for a specific trip.

This script loads a precomputed residuals table, a trained model, selects a
single trip by ID, generates a local explanation using the LIME algorithm
implemented in src.xai.lime and writes the explanation and bar plot to
disk. It assumes that the residuals table includes all the features used
by the model. Feature columns are inferred by excluding the columns
defined in TripSchema (IDs, target, prediction, residual).
"""

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import read_table, load_model, save_json, ensure_dir
from src.utils.schema import TripSchema
from src.xai.lime import explain_instance
from src.viz.plots import plot_lime_bar


def main() -> None:
    ap = argparse.ArgumentParser(description="Explain a single trip using LIME.")
    ap.add_argument(
        "--residuals_path",
        required=True,
        help="Path to residuals table containing features, predictions and residuals",
    )
    ap.add_argument(
        "--model_path",
        required=True,
        help="Path to the saved model (.joblib)",
    )
    ap.add_argument(
        "--trip_id",
        required=True,
        help="Identifier of the trip to explain",
    )
    ap.add_argument(
        "--out_dir",
        default="outputs/explanations",
        help="Directory to save explanation artifacts",
    )
    ap.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Number of perturbation samples",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top features to report",
    )
    ap.add_argument(
        "--ridge_alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength for surrogate model",
    )
    ap.add_argument(
        "--distance_metric",
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Distance metric for kernel",
    )
    ap.add_argument(
        "--kernel_width",
        type=float,
        default=None,
        help="Kernel width for exponential kernel (None => heuristic)",
    )
    ap.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed for reproducibility",
    )
    args = ap.parse_args()

    schema = TripSchema()
    out_dir = ensure_dir(args.out_dir)

    # Load data and model
    df = read_table(args.residuals_path)
    model = load_model(args.model_path)

    # Identify feature columns by excluding schema fields
    excluded = {schema.trip_id, schema.vehicle_id, schema.y, schema.yhat, schema.residual}
    feature_cols = [c for c in df.columns if c not in excluded]

    # Select the row corresponding to the requested trip
    trip_rows = df[df[schema.trip_id].astype(str) == str(args.trip_id)]
    if len(trip_rows) != 1:
        raise ValueError(f"Expected exactly 1 row for trip_id={args.trip_id}, got {len(trip_rows)}")
    x0_row = trip_rows.iloc[0]
    x0 = x0_row[feature_cols]

    # Use the entire dataset as background for perturbations (could also use train split)
    background_df = df[feature_cols].copy()

    def f_predict(Z: pd.DataFrame) -> np.ndarray:
        return model.predict(Z[feature_cols])  # type: ignore[arg-type]

    exp = explain_instance(
        trip_id=args.trip_id,
        x0=x0,
        background_df=background_df,
        feature_cols=feature_cols,
        black_box_predict=f_predict,
        n_samples=args.n_samples,
        kernel_width=args.kernel_width,
        distance_metric=args.distance_metric,
        ridge_alpha=args.ridge_alpha,
        top_k=args.top_k,
        random_state=args.random_state,
    )

    # Save explanation as JSON
    exp_path = out_dir / f"trip_{args.trip_id}_lime.json"
    save_json(
        {
            "trip_id": exp.trip_id,
            "intercept": exp.intercept,
            "kernel_width": exp.kernel_width,
            "local_r2": exp.local_r2,
            "local_rmse": exp.local_rmse,
            "top_features": exp.top_features,
            "weights": exp.weights,
        },
        exp_path,
    )

    # Save bar plot of top contributions
    plot_path = out_dir / f"trip_{args.trip_id}_lime_bar.png"
    plot_lime_bar(exp.top_features, plot_path, title=f"LIME contributions for trip {args.trip_id}")

    print(f"Saved explanation: {exp_path}")
    print(f"Saved bar plot: {plot_path}")
    print(f"local_r2={exp.local_r2:.3f}, local_rmse={exp.local_rmse:.3f}")


if __name__ == "__main__":
    main()
