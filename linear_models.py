"""
NYC Yellow Taxi fare prediction: linear models (Apurbo's part).

What this file does
-------------------
1. Loads the TLC Yellow Taxi parquet file and cleans it with the SAME rules and the
   SAME chronological 70/15/15 split as Sean's boosting notebook, so every model in
   the project is scored on identical trips.
2. Trains linear models that predict the fare in dollars:
   Linear Regression, Ridge (L2), Lasso (L1), plus a median baseline.
3. Optionally (RUN_LOGISTIC) trains logistic regression (L1 and L2) on a yes/no label:
   "is the fare >= $35?" (the point where the Uber vs taxi study says rideshare gets cheaper).
   Turn it off with --no-logistic if the team decides it is not needed.
4. Each model is trained twice:
   - Pre-trip : pickup zone, drop-off zone, pickup time only (what a rider knows in advance)
   - Post-trip: the same plus actual distance and duration (known only after the ride)
5. Reports MAE / RMSE / R2 (dollars), K-fold CV MAE, precision / recall / F1 / ROC-AUC at the
   $35 cutoff, a learning curve (train vs validation error), and the top coefficients.

Quick start
-----------
    python linear_models.py --download            # whole month (default); downloads the file if missing
    python linear_models.py --update-readme       # whole month, then fills the README results table
    python linear_models.py --quick               # first 100,000 trips only, about a minute (code check)
    python linear_models.py --synthetic           # fake demo data, no file needed

Other people's models can be put on the same $35 scale with threshold_metrics():
    from linear_models import threshold_metrics
    threshold_metrics(y_test, predicted_fares, cutoff=35)
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------------------
# Settings (change here, or override from the command line)
# --------------------------------------------------------------------------------------
DEFAULT_DATA = "data/yellow_tripdata_2026-05.parquet"
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet"
DEFAULT_SAMPLE_ROWS = 0         # 0 = whole file (default). --quick uses QUICK_SAMPLE_ROWS instead
QUICK_SAMPLE_ROWS = 100_000     # first N trips; they cover only about a day, so only for code checks
TUNE_MAX_ROWS = 200_000         # alpha / C are chosen on at most this many training trips, then the
                                # winner is refit on ALL training trips (keeps the full month fast)
# Only these columns are read from the parquet file (saves a lot of memory on 4M rows)
LOAD_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance",
                "PULocationID", "DOLocationID", "payment_type", "fare_amount"]
FARE_CUTOFF = 35.0              # dollars; used for the yes/no label
RUN_LOGISTIC = True             # set False (or pass --no-logistic) to skip logistic regression
RANDOM_STATE = 42
CV_FOLDS = 5
CV_MAX_ROWS = 50_000            # K-fold CV is run on at most this many training rows (speed)

TARGET = "fare_amount"
ZONE_COLS = ["PULocationID", "DOLocationID"]
# Time columns are one-hot encoded for linear models (hour 23 is not "bigger" than hour 1 in effect).
TIME_CAT_COLS = ["pickup_hour", "pickup_dayofweek"]
PRE_NUMERIC = ["is_weekend"]
POST_NUMERIC = PRE_NUMERIC + ["trip_distance", "trip_duration_minutes"]
FEATURE_SETS = {
    "Pre-trip": ZONE_COLS + TIME_CAT_COLS + PRE_NUMERIC,
    "Post-trip": ZONE_COLS + TIME_CAT_COLS + POST_NUMERIC,
}
NUMERIC_BY_APPROACH = {"Pre-trip": PRE_NUMERIC, "Post-trip": POST_NUMERIC}


# --------------------------------------------------------------------------------------
# Data loading and cleaning (same rules as Sean's notebook)
# --------------------------------------------------------------------------------------
def download_data(path: str = DEFAULT_DATA, url: str = DATA_URL) -> Path:
    """Download the May 2026 parquet file from the NYC TLC if it is not already on disk."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path} (about 70-80 MB)...")
    urllib.request.urlretrieve(url, path)
    return path


def load_data(path: str, sample_rows: int = DEFAULT_SAMPLE_ROWS) -> pd.DataFrame:
    """Read the parquet file. sample_rows=0 loads everything; otherwise the first N rows
    (same behaviour as Sean's notebook)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    print(f"Total rows in file: {pf.metadata.num_rows:,}")
    cols = [c for c in LOAD_COLUMNS if c in pf.schema_arrow.names]
    if not sample_rows:
        return pd.read_parquet(path, columns=cols)
    chunks, read = [], 0
    for batch in pf.iter_batches(batch_size=min(sample_rows, 100_000), columns=cols):
        chunks.append(batch.to_pandas())
        read += len(chunks[-1])
        if read >= sample_rows:
            break
    return pd.concat(chunks, ignore_index=True).head(sample_rows)


def clean_data(df: pd.DataFrame, card_only: bool = False) -> pd.DataFrame:
    """Keep plausible completed trips. Rules are copied from Sean's notebook so the rows match.

    card_only=True additionally keeps payment_type == 1 (credit card) and passenger_count > 0,
    which is what the project outline says. It is OFF by default to match Sean's rows.
    """
    data = df.copy()
    for c in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        data[c] = pd.to_datetime(data[c], errors="coerce")
    for c in ["fare_amount", "trip_distance", "PULocationID", "DOLocationID"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data["trip_duration_minutes"] = (
        (data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]).dt.total_seconds() / 60
    )
    required = ["fare_amount", "trip_distance", "tpep_pickup_datetime",
                "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]
    valid = (
        data[required].notna().all(axis=1)
        & data["fare_amount"].between(0.01, 500)
        & data["trip_distance"].between(0.01, 150)
        & data["trip_duration_minutes"].between(1, 240)
        & data["PULocationID"].between(1, 265)
        & data["DOLocationID"].between(1, 265)
    )
    if card_only:
        valid &= (data["payment_type"] == 1) & (data["passenger_count"] > 0)
    clean = data.loc[valid].copy()
    print(f"Rows before cleaning: {len(data):,} | after: {len(clean):,} | excluded: {len(data) - len(clean):,}")
    if len(clean) < 100:
        raise ValueError("Too few valid rows after cleaning. Check the file and thresholds.")
    return clean


def add_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Time features from the pickup timestamp. (pickup_month is dropped: it is constant in a
    single-month file, apart from a few trips that start on Apr 30.)"""
    out = clean.copy()
    out["pickup_hour"] = out["tpep_pickup_datetime"].dt.hour
    out["pickup_dayofweek"] = out["tpep_pickup_datetime"].dt.dayofweek
    out["is_weekend"] = (out["pickup_dayofweek"] >= 5).astype("int8")
    for c in ZONE_COLS:
        out[c] = out[c].astype("int16")
    return out


def chronological_split(df: pd.DataFrame):
    """Earliest 70% train, next 15% validation, latest 15% test (same as Sean's notebook)."""
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    n = len(df)
    a, b = int(n * 0.70), int(n * 0.85)
    train, valid, test = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    print(f"Train: {len(train):,} | Validation: {len(valid):,} | Test: {len(test):,}")
    print(f"Train dates: {train['tpep_pickup_datetime'].min()} to {train['tpep_pickup_datetime'].max()}")
    print(f"Test dates:  {test['tpep_pickup_datetime'].min()} to {test['tpep_pickup_datetime'].max()}")
    span_days = (df["tpep_pickup_datetime"].max() - df["tpep_pickup_datetime"].min()).days
    if span_days < 7:
        warnings.warn(
            f"These trips cover only about {span_days + 1} day(s). Results will not represent a full "
            "month. Run without --quick to use the whole file.", stacklevel=2)
    return train, valid, test


# --------------------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------------------
def make_preprocessor(approach: str) -> ColumnTransformer:
    """StandardScaler for numbers, one-hot for zones/hour/day. Unseen zones in later data are ignored."""
    cats = ZONE_COLS + TIME_CAT_COLS
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_BY_APPROACH[approach]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
    ])


def make_regressor(kind: str, alpha: float = 1.0) -> Pipeline:
    model = {
        "Linear": LinearRegression(),
        "Ridge (L2)": Ridge(alpha=alpha),
        "Lasso (L1)": Lasso(alpha=alpha, max_iter=5000),
    }[kind]
    return Pipeline([("prep", None), ("model", model)])  # prep filled in by caller


def make_logistic(kind: str, C: float = 1.0) -> LogisticRegression:
    """L1 or L2 logistic regression. 'balanced' class weights because $35+ trips are rare."""
    # lbfgs is fast for L2; liblinear is fast for L1 (saga was ~20x slower with the same accuracy)
    common = dict(C=C, solver="liblinear" if kind == "L1" else "lbfgs",
                  class_weight="balanced", max_iter=300)
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):  # newer scikit-learn: l1_ratio chooses the penalty
        return LogisticRegression(l1_ratio=1.0 if kind == "L1" else 0.0, **common)
    return LogisticRegression(penalty="l1" if kind == "L1" else "l2", **common)  # older versions


def build_pipeline(approach: str, estimator) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(approach)), ("model", estimator)])


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE ($)": mean_absolute_error(y_true, y_pred),
        "RMSE ($)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def threshold_metrics(y_true_fare, y_pred_fare, cutoff: float = FARE_CUTOFF) -> dict:
    """Turn ANY fare-in-dollars model into a '$cutoff or more?' classifier and score it.

    Teammates can call this on their own predictions (random forest, boosting, ...) to fill the
    same precision/recall/F1/AUC columns without retraining anything.
    The predicted fare itself is used as the score for ROC-AUC.
    """
    y_true_fare = np.asarray(y_true_fare, dtype=float)
    y_pred_fare = np.asarray(y_pred_fare, dtype=float)
    y_true = (y_true_fare >= cutoff).astype(int)
    y_hat = (y_pred_fare >= cutoff).astype(int)
    return _classification_metrics(y_true, y_hat, y_pred_fare)


def _classification_metrics(y_true, y_hat, score) -> dict:
    has_both = len(np.unique(y_true)) == 2
    return {
        "Precision": precision_score(y_true, y_hat, zero_division=0),
        "Recall": recall_score(y_true, y_hat, zero_division=0),
        "F1": f1_score(y_true, y_hat, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, score) if has_both else np.nan,
    }


# --------------------------------------------------------------------------------------
# Tuning, CV, learning curve, coefficients
# --------------------------------------------------------------------------------------
REG_ALPHAS = {
    "Ridge (L2)": [0.01, 0.1, 1, 10, 100],
    "Lasso (L1)": [0.00001, 0.0001, 0.001, 0.01, 0.1],
}
LOGIT_CS = [0.01, 0.1, 1, 10, 100]


def _tuning_slice(train: pd.DataFrame) -> pd.DataFrame:
    """A random slice of the training trips used to choose alpha / C quickly."""
    if len(train) <= TUNE_MAX_ROWS:
        return train
    return train.sample(TUNE_MAX_ROWS, random_state=RANDOM_STATE)


def tune_regressor(kind: str, approach: str, train, valid) -> tuple[float, Pipeline]:
    """Pick alpha with validation MAE only (the test set is never used for tuning).
    Alphas are compared on a slice of the training trips, then the winner is refit on all of them."""
    feats = FEATURE_SETS[approach]
    small = _tuning_slice(train)
    best_alpha, best_mae = None, np.inf
    for alpha in REG_ALPHAS[kind]:
        pipe = build_pipeline(approach, make_regressor(kind, alpha).named_steps["model"])
        pipe.fit(small[feats], small[TARGET])
        mae = mean_absolute_error(valid[TARGET], pipe.predict(valid[feats]))
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae
    best_pipe = build_pipeline(approach, make_regressor(kind, best_alpha).named_steps["model"])
    best_pipe.fit(train[feats], train[TARGET])
    return best_alpha, best_pipe


def cv_mae(kind: str, approach: str, alpha: float, train) -> float:
    """K-fold cross-validated MAE on (up to CV_MAX_ROWS of) the training period."""
    feats = FEATURE_SETS[approach]
    sub = train if len(train) <= CV_MAX_ROWS else train.sample(CV_MAX_ROWS, random_state=RANDOM_STATE)
    est = make_regressor(kind, alpha).named_steps["model"]
    pipe = build_pipeline(approach, est)
    scores = cross_val_score(
        pipe, sub[feats], sub[TARGET],
        cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_mean_absolute_error", n_jobs=1)
    return float(-scores.mean())


def learning_curve_table(approach: str, pipe: Pipeline, train, valid, fractions=(0.05, 0.1, 0.2, 0.4, 0.7, 1.0)):
    """Train on growing slices of the training period; record train MAE and validation MAE."""
    feats = FEATURE_SETS[approach]
    rows = []
    for f in fractions:
        n = max(50, int(len(train) * f))
        sub = train.sample(n, random_state=RANDOM_STATE) if n < len(train) else train
        p = clone(pipe).fit(sub[feats], sub[TARGET])
        rows.append({
            "n_train": n,
            "train_MAE": mean_absolute_error(sub[TARGET], p.predict(sub[feats])),
            "valid_MAE": mean_absolute_error(valid[TARGET], p.predict(valid[feats])),
        })
    return pd.DataFrame(rows)


def coefficient_table(pipe: Pipeline, top: int = 15) -> pd.DataFrame:
    """Largest coefficients (by absolute size) from a fitted pipeline. Inputs are scaled/one-hot, so
    numeric coefficients are per 1 standard deviation; zone and hour coefficients are vs. the baseline."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    model = pipe.named_steps["model"]
    coef = np.ravel(model.coef_)
    df = pd.DataFrame({"feature": names, "coefficient": coef})
    df["abs"] = df["coefficient"].abs()
    return df.sort_values("abs", ascending=False).head(top).drop(columns="abs").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Plots (saved as PNG; shown on screen when show=True, e.g. in the notebook)
# --------------------------------------------------------------------------------------
def _finish(fig, path, show):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


def plot_learning_curve(table, title, path=None, show=False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["n_train"], table["train_MAE"], marker="o", label="Training MAE")
    ax.plot(table["n_train"], table["valid_MAE"], marker="o", label="Validation MAE")
    ax.set_xlabel("Number of training trips")
    ax.set_ylabel("MAE ($)")
    ax.set_title(title)
    ax.legend()
    _finish(fig, path, show)


def plot_pred_vs_actual(y_true, y_pred, title, path=None, show=False, n=3000):
    import matplotlib.pyplot as plt
    y_true, y_pred = np.asarray(y_true)[:n], np.asarray(y_pred)[:n]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, alpha=0.25, s=8)
    m = max(y_true.max(), y_pred.max())
    ax.plot([0, m], [0, m], "--")
    ax.set_xlabel("Actual fare ($)")
    ax.set_ylabel("Predicted fare ($)")
    ax.set_title(title)
    _finish(fig, path, show)


def plot_coefficients(coef_df, title, path=None, show=False):
    import matplotlib.pyplot as plt
    d = coef_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(d["feature"], d["coefficient"])
    ax.set_title(title)
    ax.set_xlabel("Coefficient")
    _finish(fig, path, show)


# --------------------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------------------
def run_experiment(clean: pd.DataFrame, cutoff: float = FARE_CUTOFF, run_logistic: bool = RUN_LOGISTIC,
                   out_dir: str = "results", show_plots: bool = False):
    """Run everything on cleaned data. Returns (regression_table, logistic_table, coefficient_tables)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = add_features(clean)
    train, valid, test = chronological_split(df)

    y_test = test[TARGET].to_numpy()
    for name, part in [("train", train), ("validation", valid), ("test", test)]:
        print(f"Share of trips with fare >= ${cutoff:.0f} in {name}: {(part[TARGET] >= cutoff).mean():.2%}")

    reg_rows, coefs = [], {}
    # Baseline: always predict the median training fare
    base = DummyRegressor(strategy="median").fit(train[FEATURE_SETS["Pre-trip"]], train[TARGET])
    base_pred = base.predict(test[FEATURE_SETS["Pre-trip"]])
    reg_rows.append({"Model": "Median baseline", "Approach": "Pre-trip", "Best alpha": np.nan, "CV MAE ($)": np.nan,
                     **regression_metrics(y_test, base_pred), **threshold_metrics(y_test, base_pred, cutoff)})

    for approach in FEATURE_SETS:
        feats = FEATURE_SETS[approach]
        # Plain linear regression (no penalty)
        lin = build_pipeline(approach, LinearRegression()).fit(train[feats], train[TARGET])
        p = lin.predict(test[feats])
        reg_rows.append({"Model": "Linear", "Approach": approach, "Best alpha": np.nan,
                         "CV MAE ($)": cv_mae("Linear", approach, 0, train),
                         **regression_metrics(y_test, p), **threshold_metrics(y_test, p, cutoff)})
        for kind in ["Ridge (L2)", "Lasso (L1)"]:
            t0 = time.perf_counter()
            alpha, pipe = tune_regressor(kind, approach, train, valid)
            p = pipe.predict(test[feats])
            reg_rows.append({"Model": kind, "Approach": approach, "Best alpha": alpha,
                             "CV MAE ($)": cv_mae(kind, approach, alpha, train),
                             **regression_metrics(y_test, p), **threshold_metrics(y_test, p, cutoff)})
            coefs[(kind, approach)] = coefficient_table(pipe)
            print(f"{kind} | {approach}: alpha={alpha}, done in {time.perf_counter() - t0:.1f}s")
            tag = f"{kind.split()[0].lower()}_{approach.lower().replace('-', '')}"
            plot_pred_vs_actual(y_test, p, f"{kind} | {approach}: Predicted vs Actual",
                                out / "plots" / f"pred_vs_actual_{tag}.png", show_plots)
            plot_coefficients(coefs[(kind, approach)], f"{kind} | {approach}: top coefficients",
                              out / "plots" / f"coefficients_{tag}.png", show_plots)
            if kind == "Ridge (L2)":
                lc = learning_curve_table(approach, pipe, train, valid)
                lc.to_csv(out / f"learning_curve_{approach.lower().replace('-', '')}.csv", index=False)
                plot_learning_curve(lc, f"Ridge | {approach}: learning curve",
                                    out / "plots" / f"learning_curve_{approach.lower().replace('-', '')}.png", show_plots)

    reg_df = pd.DataFrame(reg_rows).sort_values("MAE ($)").reset_index(drop=True)
    reg_df.to_csv(out / "linear_regression_results.csv", index=False)

    log_df = pd.DataFrame()
    if run_logistic:
        log_df = run_logistic_models(train, valid, test, cutoff, out)
    for (kind, approach), table in coefs.items():
        table.to_csv(out / f"coefficients_{kind.split()[0].lower()}_{approach.lower().replace('-', '')}.csv", index=False)
    return reg_df, log_df, coefs


def run_logistic_models(train, valid, test, cutoff, out: Path) -> pd.DataFrame:
    """Logistic regression (L1 and L2) predicting 'fare >= cutoff'. Optional extra from the outline."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ytr = (train[TARGET] >= cutoff).astype(int)
    yva = (valid[TARGET] >= cutoff).astype(int)
    yte = (test[TARGET] >= cutoff).astype(int)
    if ytr.sum() < 20 or yte.sum() < 5:
        print(f"WARNING: too few trips at ${cutoff:.0f}+ (train={int(ytr.sum())}, test={int(yte.sum())}). "
              "Skipping logistic regression. Use more data (run without --quick) or a lower --cutoff.")
        return pd.DataFrame()
    rows = []
    small = _tuning_slice(train)
    ysm = (small[TARGET] >= cutoff).astype(int)
    for approach, feats in FEATURE_SETS.items():
        for kind in ["L1", "L2"]:
            best_ap, C = -np.inf, None
            for c in LOGIT_CS:  # choose C by validation average precision (good for rare classes)
                cand = build_pipeline(approach, make_logistic(kind, c))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cand.fit(small[feats], ysm)
                ap = average_precision_score(yva, cand.predict_proba(valid[feats])[:, 1])
                if ap > best_ap:
                    best_ap, C = ap, c
            pipe = build_pipeline(approach, make_logistic(kind, C))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(train[feats], ytr)
            proba = pipe.predict_proba(test[feats])[:, 1]
            hat = (proba >= 0.5).astype(int)
            rows.append({"Model": f"Logistic ({kind})", "Approach": approach, "Best C": C,
                         "MAE ($)": np.nan, "RMSE ($)": np.nan, "R2": np.nan,
                         **_classification_metrics(yte.to_numpy(), hat, proba)})
            tag = f"logistic_{kind.lower()}_{approach.lower().replace('-', '')}"
            coefficient_table(pipe).to_csv(out / f"coefficients_{tag}.csv", index=False)
            print(f"Logistic {kind} | {approach}: C={C}")
    log_df = pd.DataFrame(rows)
    log_df.to_csv(out / "logistic_results.csv", index=False)
    return log_df


# --------------------------------------------------------------------------------------
# Demo data (only for trying the script when the real file is not available)
# --------------------------------------------------------------------------------------
def make_synthetic(n: int = 60_000, seed: int = 0) -> pd.DataFrame:
    """Fake taxi trips with the same columns as the TLC file. NOT real data; for testing only."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-05-01")
    pickup = start + pd.to_timedelta(np.sort(rng.uniform(0, 31 * 24 * 3600, n)), unit="s")
    dist = np.clip(rng.lognormal(0.9, 0.9, n), 0.1, 40)
    dur = np.clip(dist * rng.uniform(3, 6, n) + rng.uniform(2, 8, n), 1.5, 200)
    pu = rng.integers(1, 266, n)
    do = rng.integers(1, 266, n)
    airport = np.isin(do, [132, 138]) | np.isin(pu, [132, 138])
    fare = 3 + 0.7 * 5 * dist + airport * 8 + rng.normal(0, 1.2, n)
    return pd.DataFrame({
        "VendorID": 2, "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": pickup + pd.to_timedelta(dur, unit="m"),
        "passenger_count": rng.integers(1, 4, n), "trip_distance": dist, "RatecodeID": 1,
        "store_and_fwd_flag": "N", "PULocationID": pu, "DOLocationID": do, "payment_type": 1,
        "fare_amount": np.clip(fare, 3, None),
    })


# --------------------------------------------------------------------------------------
# README results (filled in by --update-readme)
# --------------------------------------------------------------------------------------
README_START = "<!-- LINEAR-RESULTS:START -->"
README_END = "<!-- LINEAR-RESULTS:END -->"


def _md_table(df: pd.DataFrame, fmt: dict) -> str:
    """Tiny markdown table writer (avoids needing the 'tabulate' package)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, str):
                cells.append(v)
            elif pd.isna(v):
                cells.append("-")
            else:
                cells.append(fmt.get(c, "{:g}").format(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_readme(reg_df, log_df, clean, sample_rows, cutoff, readme="README.md") -> None:
    """Replace the block between the LINEAR-RESULTS markers in the README with this run's tables."""
    path = Path(readme)
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise SystemExit(f"{readme} has no {README_START} ... {README_END} block to fill.")
    d0 = clean["tpep_pickup_datetime"].min().date()
    d1 = clean["tpep_pickup_datetime"].max().date()
    scope = "the whole month" if not sample_rows else f"the first {sample_rows:,} trips in the file"
    reg = reg_df[["Model", "Approach", "MAE ($)", "RMSE ($)", "R2", "CV MAE ($)"]]
    parts = [
        README_START,
        f"_Run on {scope}: {len(clean):,} trips after cleaning, pickups from {d0} to {d1}. "
        f"Test = the latest 15% of trips. Generated by `python linear_models.py --update-readme`._",
        "",
        "**Dollar-fare models** (lower MAE/RMSE is better; CV MAE is 5-fold on the training period)",
        "",
        _md_table(reg, {"MAE ($)": "{:.2f}", "RMSE ($)": "{:.2f}", "R2": "{:.3f}", "CV MAE ($)": "{:.2f}"}),
    ]
    if len(log_df):
        cls = log_df[["Model", "Approach", "Precision", "Recall", "F1", "ROC-AUC"]]
        parts += ["", f"**Logistic regression: is the fare ${cutoff:.0f} or more?**", "",
                  _md_table(cls, {k: "{:.3f}" for k in ["Precision", "Recall", "F1", "ROC-AUC"]})]
    parts.append(README_END)
    head, rest = text.split(README_START, 1)
    tail = rest.split(README_END, 1)[1]
    path.write_text(head + "\n".join(parts) + tail, encoding="utf-8")
    print(f"README results updated in {readme}")


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Linear and logistic models for NYC Yellow Taxi fares")
    ap.add_argument("--data", default=DEFAULT_DATA, help="path to yellow_tripdata_2026-05.parquet")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE_ROWS,
                    help="rows to use; 0 = whole file (default)")
    ap.add_argument("--quick", action="store_true",
                    help=f"only the first {QUICK_SAMPLE_ROWS:,} trips (about a day; for checking the code)")
    ap.add_argument("--cutoff", type=float, default=FARE_CUTOFF, help="dollar cutoff for the yes/no label")
    ap.add_argument("--no-logistic", action="store_true", help="skip logistic regression")
    ap.add_argument("--card-only", action="store_true", help="keep only card payments and passenger_count > 0")
    ap.add_argument("--download", action="store_true", help="download the parquet file if it is missing")
    ap.add_argument("--synthetic", action="store_true", help="use fake demo data instead of the real file")
    ap.add_argument("--out", default="results", help="folder for result tables and plots")
    ap.add_argument("--update-readme", action="store_true",
                    help="write this run's result tables into README.md (not allowed with --synthetic)")
    args = ap.parse_args(argv)
    if args.quick:
        args.sample = QUICK_SAMPLE_ROWS
    if args.update_readme and args.synthetic:
        sys.exit("--update-readme is not allowed with --synthetic: fake results must not go in the README.")

    import matplotlib
    matplotlib.use("Agg")

    if args.synthetic:
        print("Using SYNTHETIC demo data. Results are not real.")
        raw = make_synthetic()
    else:
        if not Path(args.data).exists():
            if args.download:
                download_data(args.data)
            else:
                sys.exit(f"Data file not found: {args.data}\n"
                         "Put the parquet file there, or re-run with --download, or use --synthetic.")
        raw = load_data(args.data, args.sample)

    clean = clean_data(raw, card_only=args.card_only)
    reg_df, log_df, _ = run_experiment(clean, args.cutoff, not args.no_logistic, args.out)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== Dollar-fare models (test period) ===")
    print(reg_df.round(3).to_string(index=False))
    if len(log_df):
        print(f"\n=== Logistic regression: fare >= ${args.cutoff:.0f} (test period) ===")
        print(log_df.drop(columns=["MAE ($)", "RMSE ($)", "R2"]).round(3).to_string(index=False))
    print(f"\nTables and plots saved in: {args.out}/")
    if args.update_readme:
        update_readme(reg_df, log_df, clean, args.sample, args.cutoff)


if __name__ == "__main__":
    main()
