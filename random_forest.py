"""
NYC Yellow Taxi fare prediction: tree ensemble models (Zhanel's part).

What this file does
-------------------
1. Loads, cleans and splits the May 2026 trips with the SAME functions as linear_models.py
   (same cleaning rules, same chronological 70/15/15 split), so every model in the project
   is scored on identical trips.
2. Trains tree models that predict the fare in dollars:
   a single Decision Tree, Bagging (bagged decision trees) and a Random Forest,
   plus the same median baseline.
3. Each model is trained twice, like the rest of the team:
   - Pre-trip : pickup zone, drop-off zone, pickup time only (what a rider knows in advance)
   - Post-trip: the same plus actual distance and duration (known only after the ride)
4. Tunes the random forest (min_samples_leaf, max_features) on the VALIDATION period only,
   then scores every model once on the TEST period.
5. Reports MAE / RMSE / R2 (dollars), training MAE (to check overfitting), 5-fold CV MAE,
   precision / recall / F1 / ROC-AUC at the $35 cutoff, a learning curve, predicted-vs-actual
   plots and feature importances.

Quick start
-----------
    python random_forest.py --download            # whole month (default); downloads the file if missing
    python random_forest.py --update-readme       # whole month, then fills the README results table
    python random_forest.py --quick               # first 100,000 trips only (code check)
    python random_forest.py --synthetic           # fake demo data, no file needed

Why trees use the zone IDs as numbers here
------------------------------------------
Tree models split on thresholds, so they can carve the zone ID range into groups without one-hot
encoding (one-hot with ~260 zones x 2 on ~2.7 million trips would be very slow for forests).
The inputs are exactly the same columns the linear and boosting models use.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

import linear_models as lm  # shared loading, cleaning, features, split, metrics and plots

# --------------------------------------------------------------------------------------
# Settings (change here, or override from the command line)
# --------------------------------------------------------------------------------------
RANDOM_STATE = lm.RANDOM_STATE
TARGET = lm.TARGET                       # fare_amount, same as the rest of the team
N_TREES = 100                            # trees in the final bagging and random forest models
TUNE_TREES = 50                          # trees per candidate while tuning (faster)
TREE_MAX_SAMPLES = 200_000               # each tree is trained on a bootstrap sample of at most this
                                         # many trips; keeps the whole month fast and memory-safe
TUNE_MAX_ROWS = lm.TUNE_MAX_ROWS         # random forest settings are compared on at most this many trips
CV_FOLDS = lm.CV_FOLDS
CV_MAX_ROWS = lm.CV_MAX_ROWS             # K-fold CV is run on at most this many training trips
IMPORTANCE_ROWS = 20_000                 # test trips used for permutation importance

PRE_FEATURES = ["PULocationID", "DOLocationID", "pickup_hour", "pickup_dayofweek", "is_weekend"]
FEATURE_SETS = {
    "Pre-trip": PRE_FEATURES,
    "Post-trip": PRE_FEATURES + ["trip_distance", "trip_duration_minutes"],
}

# Random forest settings compared on the validation period
RF_GRID = [
    {"min_samples_leaf": leaf, "max_features": feat}
    for leaf in [5, 20]
    for feat in [0.5, 1.0]
]


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def _max_samples(n_rows: int):
    """Bootstrap sample size per tree: all rows if small, else TREE_MAX_SAMPLES."""
    return None if n_rows <= TREE_MAX_SAMPLES else TREE_MAX_SAMPLES


def make_model(kind: str, n_rows: int, params: dict | None = None, n_trees: int | None = None):
    """Build one regressor. n_rows = number of training rows it will be fit on."""
    params = params or {}
    n_trees = n_trees or N_TREES
    if kind == "Decision tree":
        return DecisionTreeRegressor(min_samples_leaf=20, random_state=RANDOM_STATE)
    if kind == "Bagging":
        ms = _max_samples(n_rows)
        return BaggingRegressor(
            estimator=DecisionTreeRegressor(min_samples_leaf=5),
            n_estimators=n_trees, max_samples=1.0 if ms is None else ms,
            bootstrap=True, n_jobs=-1, random_state=RANDOM_STATE)
    if kind == "Random forest":
        return RandomForestRegressor(
            n_estimators=n_trees, max_samples=_max_samples(n_rows),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_features=params.get("max_features", 0.5),
            n_jobs=-1, random_state=RANDOM_STATE)
    raise ValueError(kind)


def tune_random_forest(approach: str, train, valid) -> tuple[dict, pd.DataFrame]:
    """Pick random forest settings by VALIDATION MAE (the test set is never used for tuning)."""
    feats = FEATURE_SETS[approach]
    small = train if len(train) <= TUNE_MAX_ROWS else train.sample(TUNE_MAX_ROWS, random_state=RANDOM_STATE)
    rows = []
    for params in RF_GRID:
        t0 = time.perf_counter()
        rf = make_model("Random forest", len(small), params, n_trees=TUNE_TREES).fit(small[feats], small[TARGET])
        rows.append({**params, "Validation MAE ($)": mean_absolute_error(valid[TARGET], rf.predict(valid[feats])),
                     "Seconds": time.perf_counter() - t0})
    table = pd.DataFrame(rows).sort_values("Validation MAE ($)").reset_index(drop=True)
    best = table.iloc[0]
    best_params = {"min_samples_leaf": int(best["min_samples_leaf"]), "max_features": float(best["max_features"])}
    print(f"Random forest | {approach}: best settings {best_params} "
          f"(validation MAE ${best['Validation MAE ($)']:.2f})")
    return best_params, table


def cv_mae(kind: str, approach: str, params: dict | None, train) -> float:
    """K-fold cross-validated MAE on (up to CV_MAX_ROWS of) the training period."""
    feats = FEATURE_SETS[approach]
    sub = train if len(train) <= CV_MAX_ROWS else train.sample(CV_MAX_ROWS, random_state=RANDOM_STATE)
    model = make_model(kind, len(sub), params)
    scores = cross_val_score(model, sub[feats], sub[TARGET],
                             cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                             scoring="neg_mean_absolute_error", n_jobs=1)
    return float(-scores.mean())


def train_mae(model, train, feats, n: int = 200_000) -> float:
    """MAE on (a sample of) the training trips; compare with test MAE to spot overfitting."""
    sub = train if len(train) <= n else train.sample(n, random_state=RANDOM_STATE)
    return mean_absolute_error(sub[TARGET], model.predict(sub[feats]))


def learning_curve_table(approach: str, params: dict, train, valid,
                         fractions=(0.02, 0.05, 0.1, 0.3, 1.0)) -> pd.DataFrame:
    """Train on growing slices of the training period; record train MAE and validation MAE."""
    feats = FEATURE_SETS[approach]
    rows = []
    for f in fractions:
        n = max(200, int(len(train) * f))
        sub = train.sample(n, random_state=RANDOM_STATE) if n < len(train) else train
        rf = make_model("Random forest", len(sub), params, n_trees=TUNE_TREES).fit(sub[feats], sub[TARGET])
        rows.append({"n_train": n,
                     "train_MAE": train_mae(rf, sub, feats, n=50_000),
                     "valid_MAE": mean_absolute_error(valid[TARGET], rf.predict(valid[feats]))})
    return pd.DataFrame(rows)


def importance_table(model, test, feats) -> pd.DataFrame:
    """Two views of feature importance for a fitted forest:
    - impurity: how much each feature reduces error inside the trees (fast, built in)
    - permutation: how much test MAE gets worse when the feature is shuffled (more reliable)"""
    sub = test if len(test) <= IMPORTANCE_ROWS else test.sample(IMPORTANCE_ROWS, random_state=RANDOM_STATE)
    perm = permutation_importance(model, sub[feats], sub[TARGET], n_repeats=3,
                                  scoring="neg_mean_absolute_error", random_state=RANDOM_STATE, n_jobs=1)
    return (pd.DataFrame({"feature": feats,
                          "impurity_importance": model.feature_importances_,
                          "permutation_MAE_increase ($)": perm.importances_mean})
            .sort_values("permutation_MAE_increase ($)", ascending=False).reset_index(drop=True))


def plot_importance(table, title, path=None, show=False):
    import matplotlib.pyplot as plt
    d = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(d["feature"], d["permutation_MAE_increase ($)"])
    ax.set_xlabel("Increase in test MAE when shuffled ($)")
    ax.set_title(title)
    lm._finish(fig, path, show)


# --------------------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------------------
def run_experiment(clean: pd.DataFrame, cutoff: float = lm.FARE_CUTOFF, out_dir: str = "results",
                   show_plots: bool = False):
    """Run everything on cleaned data. Returns (results_table, tuning_tables, importance_tables, models)."""
    out = Path(out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    df = lm.add_features(clean)
    train, valid, test = lm.chronological_split(df)
    y_test = test[TARGET].to_numpy()

    rows, tuning, importances, fitted = [], {}, {}, {}
    base = DummyRegressor(strategy="median").fit(train[PRE_FEATURES], train[TARGET])
    base_pred = base.predict(test[PRE_FEATURES])
    rows.append({"Model": "Median baseline", "Approach": "Pre-trip", "Settings": "-",
                 "Train MAE ($)": train_mae(base, train, PRE_FEATURES), "CV MAE ($)": np.nan,
                 **lm.regression_metrics(y_test, base_pred), **lm.threshold_metrics(y_test, base_pred, cutoff),
                 "Train seconds": np.nan})

    for approach, feats in FEATURE_SETS.items():
        tag = approach.lower().replace("-", "")
        best_params, tuning[approach] = tune_random_forest(approach, train, valid)
        tuning[approach].to_csv(out / f"rf_tuning_{tag}.csv", index=False)

        for kind in ["Decision tree", "Bagging", "Random forest"]:
            params = best_params if kind == "Random forest" else None
            t0 = time.perf_counter()
            model = make_model(kind, len(train), params).fit(train[feats], train[TARGET])
            secs = time.perf_counter() - t0
            pred = model.predict(test[feats])
            fitted[(kind, approach)] = model
            settings = (f"leaf={params['min_samples_leaf']}, max_features={params['max_features']}"
                        if params else "-")
            rows.append({"Model": kind, "Approach": approach, "Settings": settings,
                         "Train MAE ($)": train_mae(model, train, feats),
                         "CV MAE ($)": cv_mae(kind, approach, params, train),
                         **lm.regression_metrics(y_test, pred), **lm.threshold_metrics(y_test, pred, cutoff),
                         "Train seconds": secs})
            print(f"{kind} | {approach}: test MAE ${rows[-1]['MAE ($)']:.2f}, trained in {secs:.1f}s")
            ktag = kind.lower().replace(" ", "_")
            lm.plot_pred_vs_actual(y_test, pred, f"{kind} | {approach}: Predicted vs Actual",
                                   out / "plots" / f"pred_vs_actual_{ktag}_{tag}.png", show_plots)

        rf = fitted[("Random forest", approach)]
        importances[approach] = importance_table(rf, test, feats)
        importances[approach].to_csv(out / f"rf_importance_{tag}.csv", index=False)
        plot_importance(importances[approach], f"Random forest | {approach}: feature importance",
                        out / "plots" / f"rf_importance_{tag}.png", show_plots)
        lc = learning_curve_table(approach, best_params, train, valid)
        lc.to_csv(out / f"rf_learning_curve_{tag}.csv", index=False)
        lm.plot_learning_curve(lc, f"Random forest | {approach}: learning curve",
                               out / "plots" / f"rf_learning_curve_{tag}.png", show_plots)

    results = pd.DataFrame(rows).sort_values("MAE ($)").reset_index(drop=True)
    results.to_csv(out / "random_forest_results.csv", index=False)
    return results, tuning, importances, fitted


# --------------------------------------------------------------------------------------
# README results (filled in by --update-readme)
# --------------------------------------------------------------------------------------
README_START = "<!-- RF-RESULTS:START -->"
README_END = "<!-- RF-RESULTS:END -->"


def update_readme(results, clean, sample_rows, readme="README.md") -> None:
    """Replace the block between the RF-RESULTS markers in the README with this run's table."""
    path = Path(readme)
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise SystemExit(f"{readme} has no {README_START} ... {README_END} block to fill.")
    d0 = clean["tpep_pickup_datetime"].min().date()
    d1 = clean["tpep_pickup_datetime"].max().date()
    scope = "the whole month" if not sample_rows else f"the first {sample_rows:,} trips in the file"
    table = results[["Model", "Approach", "MAE ($)", "RMSE ($)", "R2", "Train MAE ($)", "CV MAE ($)"]]
    block = "\n".join([
        README_START,
        f"_Run on {scope}: {len(clean):,} trips after cleaning, pickups from {d0} to {d1}. "
        f"Test = the latest 15% of trips. Generated by `python random_forest.py --update-readme`._",
        "",
        "**Dollar-fare models** (lower MAE/RMSE is better; Train MAE vs MAE shows overfitting; "
        "CV MAE is 5-fold on the training period)",
        "",
        lm._md_table(table, {"MAE ($)": "{:.2f}", "RMSE ($)": "{:.2f}", "R2": "{:.3f}",
                             "Train MAE ($)": "{:.2f}", "CV MAE ($)": "{:.2f}"}),
        README_END,
    ])
    head, rest = text.split(README_START, 1)
    tail = rest.split(README_END, 1)[1]
    path.write_text(head + block + tail, encoding="utf-8")
    print(f"README results updated in {readme}")


# --------------------------------------------------------------------------------------
def main(argv=None):
    global N_TREES
    ap = argparse.ArgumentParser(description="Decision tree, bagging and random forest for NYC Yellow Taxi fares")
    ap.add_argument("--data", default=lm.DEFAULT_DATA, help="path to yellow_tripdata_2026-05.parquet")
    ap.add_argument("--sample", type=int, default=lm.DEFAULT_SAMPLE_ROWS, help="rows to use; 0 = whole file (default)")
    ap.add_argument("--quick", action="store_true", help=f"only the first {lm.QUICK_SAMPLE_ROWS:,} trips (code check)")
    ap.add_argument("--trees", type=int, default=N_TREES, help="trees in the final bagging and random forest models")
    ap.add_argument("--cutoff", type=float, default=lm.FARE_CUTOFF, help="dollar cutoff for the $35+ metrics")
    ap.add_argument("--card-only", action="store_true", help="keep only card payments and passenger_count > 0")
    ap.add_argument("--all-dates", action="store_true", help="do not drop trips dated outside May 2026 (not recommended)")
    ap.add_argument("--download", action="store_true", help="download the parquet file if it is missing")
    ap.add_argument("--synthetic", action="store_true", help="use fake demo data instead of the real file")
    ap.add_argument("--out", default="results", help="folder for result tables and plots")
    ap.add_argument("--update-readme", action="store_true", help="write this run's results table into README.md")
    args = ap.parse_args(argv)
    if args.quick:
        args.sample = lm.QUICK_SAMPLE_ROWS
    if args.update_readme and args.synthetic:
        sys.exit("--update-readme is not allowed with --synthetic: fake results must not go in the README.")
    N_TREES = args.trees

    import matplotlib
    matplotlib.use("Agg")

    if args.synthetic:
        print("Using SYNTHETIC demo data. Results are not real.")
        raw = lm.make_synthetic()
    else:
        if not Path(args.data).exists():
            if args.download:
                lm.download_data(args.data)
            else:
                sys.exit(f"Data file not found: {args.data}\n"
                         "Put the parquet file there, or re-run with --download, or use --synthetic.")
        raw = lm.load_data(args.data, args.sample)

    clean = lm.clean_data(raw, card_only=args.card_only, month_only=not args.all_dates)
    results, _, importances, _ = run_experiment(clean, args.cutoff, args.out)

    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("\n=== Tree models (test period) ===")
    print(results.round(3).to_string(index=False))
    for approach, table in importances.items():
        print(f"\n=== Random forest feature importance | {approach} ===")
        print(table.round(3).to_string(index=False))
    print(f"\nTables and plots saved in: {args.out}/")
    if args.update_readme:
        update_readme(results, clean, args.sample)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
