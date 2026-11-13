# MIS545-NYCTaxi

**Predictive Modeling of Yellow Taxi Prices in NYC** (MIS 545, Team: The Office)

We predict the fare of a NYC Yellow Taxi trip from the trip's details, using the NYC TLC trip records for **May 2026** (4,090,836 trips).

Each model is trained two ways:
- **Pre-trip**: only what a rider knows in advance (pickup zone, drop-off zone, pickup time).
- **Post-trip**: the same, plus actual distance and duration (only known after the ride).

## What is in this repo

| Path | What it is | Who |
|---|---|---|
| `notebooks/boosting/` | XGBoost, LightGBM, CatBoost notebook | Sean |
| `linear_models.py` | Linear, Ridge, Lasso and logistic regression (runnable script) | Apurbo |
| `notebooks/linear/` | Notebook version of the linear models (calls `linear_models.py`) | Apurbo |
| `data/` | Put the dataset here (see below) | |
| `results/` | Tables and plots are saved here when you run the code (not saved to git) | |

## The data

File: `yellow_tripdata_2026-05.parquet` from the [NYC TLC trip record page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

Put it in the `data/` folder, so the path is `data/yellow_tripdata_2026-05.parquet`.
If it is not there, add `--download` when you run the script and it will fetch it for you.

## Option 1: Run the Python script (easiest to repeat)

1. Get the code (only the first time):
   ```
   git clone https://github.com/nataperezzr/MIS545-NYCTaxi.git
   cd MIS545-NYCTaxi
   ```
   Already cloned? Get the newest version with `git pull`.

2. Install what it needs (only the first time):
   ```
   pip install -r requirements.txt
   ```

3. Run it:
   ```
   python linear_models.py --download
   ```
   This downloads the data if needed and runs the **whole month** (about 4 million trips). It takes roughly 15 to 25 minutes and needs a few GB of memory.

   Just want to check that it works? Use the quick version (first 100,000 trips, about a minute):
   ```
   python linear_models.py --quick
   ```

   To run the whole month and also fill in the results table in this README:
   ```
   python linear_models.py --update-readme
   ```

4. Look in the `results/` folder for the tables (`.csv`) and plots (`plots/`).

No data yet and just want to check that it works? Run `python linear_models.py --synthetic` (fake demo data, results are not real).

### Useful options

| Option | What it does |
|---|---|
| `--quick` | Only the first 100,000 trips (about a day and a half). For checking the code. |
| `--sample N` | Use the first N trips. `0` (the default) = whole file. |
| `--cutoff 35` | Dollar cutoff for the yes/no ("fare is this much or more?") label. |
| `--no-logistic` | Skip logistic regression. |
| `--card-only` | Keep only card payments with at least 1 passenger. |
| `--out results` | Folder for results. |

## Option 2: Run the notebook

**In Google Colab (no install needed)**
1. Go to [colab.research.google.com](https://colab.research.google.com), choose **File > Open notebook > GitHub**, and paste `nataperezzr/MIS545-NYCTaxi`.
2. Open `notebooks/linear/NYC_Yellow_Taxi_Linear_and_Logistic.ipynb`.
3. Click **Runtime > Run all**. The first cell downloads this repo, and the data cell downloads the dataset (or asks you to upload it).

**On your own computer**
1. Clone the repo and run `pip install -r requirements.txt` (see Option 1, steps 1 and 2).
2. Run `jupyter notebook` and open `notebooks/linear/NYC_Yellow_Taxi_Linear_and_Logistic.ipynb`.
3. Choose **Run all**.

Settings (sample size, dollar cutoff, logistic on/off) are in the notebook's "Settings" cell. By default it runs the whole month, which takes a while; set `SAMPLE_ROWS = 100_000` there for a quick check.

## Comparing everyone's models on the same scale

All models in the project use the same cleaning rules and the same time-based split (earliest 70% train, next 15% validation, latest 15% test), so scores are comparable.

To also score any dollar-fare model as a "fare is $35 or more?" classifier, no retraining needed:
```python
from linear_models import threshold_metrics
threshold_metrics(y_test, predicted_fares, cutoff=35)   # precision, recall, F1, ROC-AUC
```

## Results

**Preliminary.** These come from the first 100,000 trips in the file, which cover only about a day and a half (Apr 30 evening to May 2 early morning; the test period is the evening of May 1). They show that the code works, not how the models do on the whole month. We will replace them with full-month numbers.

Test-period MAE in dollars (lower is better):

### Boosting models (Sean)

| Model | Approach | MAE ($) | RMSE ($) | R2 |
|---|---|---|---|---|
| XGBoost | Post-trip | 0.95 | 3.90 | 0.938 |
| LightGBM | Post-trip | 0.97 | 3.49 | 0.951 |
| CatBoost | Post-trip | 1.00 | 3.65 | 0.946 |
| XGBoost | Pre-trip | 3.77 | 7.14 | 0.794 |
| LightGBM | Pre-trip | 3.79 | 6.91 | 0.807 |
| CatBoost | Pre-trip | 4.43 | 8.24 | 0.725 |
| Median baseline | Pre-trip | 9.18 | 16.31 | -0.075 |

Knowing the actual distance and duration (post-trip) cuts the error by about $2.80 to $3.40 per trip. The three algorithms are very close on post-trip, so the gaps between them are small.

### Linear models (Apurbo)

Not run on the whole month yet. After running the script on the real file, this table fills itself in:
```
python linear_models.py --update-readme
```

<!-- LINEAR-RESULTS:START -->
_Not filled in yet._
<!-- LINEAR-RESULTS:END -->

### Random forest (Zhanel)

_To be added._

## Notes
- The raw "May 2026" file contains some trips with wrong dates (for example 2009). `linear_models.py` drops every trip whose pickup is outside May 2026 (`--all-dates` turns this off). Any notebook that uses this file should do the same, so all models are scored on the same trips.
- The quick 100,000-trip sample covers only about a day and a half (the file is ordered by time), so those numbers are only for checking that the code works. Use the whole file for results we report.
- One month of data cannot tell us about other seasons.
