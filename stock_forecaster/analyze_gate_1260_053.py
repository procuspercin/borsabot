import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from pathlib import Path
import time

INPUT = Path("data/base_30d_regime_predictions_enriched.csv")
OUTPUT = Path("data/gate_1260_053_regime_diagnostics.csv")

WINDOW = 1260
THRESHOLD = 0.53
MIN_HISTORY = 300
TARGET_HORIZON = 30

LEVELS = [
    ("FULL", [
        "MarketRegime",
        "Momentum20Bucket",
        "Momentum60Bucket",
        "VolatilityBucket",
        "SMA200Bucket",
    ]),
    ("TREND_VOL_M20", [
        "MarketRegime",
        "VolatilityBucket",
        "Momentum20Bucket",
    ]),
    ("TREND_VOL", [
        "MarketRegime",
        "VolatilityBucket",
    ]),
    ("M20_VOL", [
        "Momentum20Bucket",
        "VolatilityBucket",
    ]),
    ("M20", ["Momentum20Bucket"]),
    ("MARKET", ["MarketRegime"]),
]

def safe_auc(y, p):
    if len(y) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)

def metrics(g):
    if len(g) == 0:
        return np.nan, np.nan, np.nan

    y = g["actual"].to_numpy()
    p = g["probability"].to_numpy()
    pred = (p >= .50).astype(int)

    auc = safe_auc(y, p)

    if len(np.unique(y)) < 2:
        bal = np.nan
    else:
        bal = balanced_accuracy_score(y, pred)

    acc = np.mean(pred == y)

    return auc, bal, acc

print("=" * 100)
print("1260 / 0.53 GATE — REGIME DIAGNOSTICS")
print("=" * 100)

t0 = time.time()

df = pd.read_csv(INPUT, parse_dates=["Date"])
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

regime_cols = [
    "MarketRegime",
    "Momentum20Bucket",
    "Momentum60Bucket",
    "VolatilityBucket",
    "SMA200Bucket",
]

for c in regime_cols:
    df[c] = df[c].astype("string").fillna("NA")

dates = (
    df["Date"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

date_map = {d: i for i, d in enumerate(dates)}

df["_date_idx"] = df["Date"].map(date_map)
df["_available_idx"] = df["_date_idx"] + TARGET_HORIZON

by_date = {
    int(di): g.index.to_numpy()
    for di, g in df.groupby("_date_idx")
}

gate_records = []

for di in range(len(dates)):

    today_ids = by_date.get(di)

    if today_ids is None:
        continue

    hist_start = max(0, di - WINDOW)

    hist = df[
        (df["_available_idx"] <= di) &
        (df["_date_idx"] >= hist_start)
    ]

    maps = {}

    if len(hist) >= MIN_HISTORY:

        for level_name, cols in LEVELS:

            level_map = {}

            for key, g in hist.groupby(
                cols,
                observed=True,
                sort=False
            ):

                if len(g) < MIN_HISTORY:
                    continue

                if not isinstance(key, tuple):
                    key = (key,)

                auc, bal, acc = metrics(g)

                level_map[
                    tuple(str(x) for x in key)
                ] = {
                    "hist_samples": len(g),
                    "hist_auc": auc,
                    "hist_balacc": bal,
                }

            maps[level_name] = (cols, level_map)

    today = df.loc[today_ids]

    for rid, row in today.iterrows():

        chosen_level = "NONE"
        hist_samples = 0
        hist_auc = np.nan
        hist_bal = np.nan

        for level_name, cols in LEVELS:

            if level_name not in maps:
                continue

            _, level_map = maps[level_name]

            key = tuple(str(row[c]) for c in cols)

            if key in level_map:
                m = level_map[key]

                chosen_level = level_name
                hist_samples = m["hist_samples"]
                hist_auc = m["hist_auc"]
                hist_bal = m["hist_balacc"]
                break

        active = (
            pd.notna(hist_auc)
            and hist_auc >= THRESHOLD
            and pd.notna(hist_bal)
            and hist_bal >= .50
        )

        gate_records.append({
            "_row_id": rid,
            "GateActive": active,
            "GateLevel": chosen_level,
            "HistSamples": hist_samples,
            "HistAUC": hist_auc,
            "HistBalAcc": hist_bal,
        })

    if di % 300 == 0:
        print(
            f"{di}/{len(dates)} | "
            f"{dates.iloc[di].date()}"
        )

gate = pd.DataFrame(gate_records).set_index("_row_id")

df = df.join(gate)

selected = df[df["GateActive"] == True].copy()

print()
print(f"Gate selected: {len(selected):,}/{len(df):,} "
      f"(%{100*len(selected)/len(df):.2f})")

# ----------------------------------------------------------
# YEAR SUMMARY
# ----------------------------------------------------------

print()
print("=" * 100)
print("YEAR SUMMARY")
print("=" * 100)

year_rows = []

for year, g in selected.groupby("Year"):

    auc, bal, acc = metrics(g)

    year_rows.append({
        "Year": year,
        "Samples": len(g),
        "AUC": auc,
        "BalancedAccuracy": bal,
        "Accuracy": acc,
    })

year_df = pd.DataFrame(year_rows)

print(
    year_df.to_string(
        index=False,
        formatters={
            "AUC": lambda x: f"{x:.4f}",
            "BalancedAccuracy": lambda x: f"%{x*100:.2f}",
            "Accuracy": lambda x: f"%{x*100:.2f}",
        }
    )
)

# ----------------------------------------------------------
# REGIME DIAGNOSTICS
# ----------------------------------------------------------

diagnostic_rows = []

GROUPINGS = {
    "Market": ["MarketRegime"],
    "Momentum20": ["Momentum20Bucket"],
    "Momentum60": ["Momentum60Bucket"],
    "Volatility": ["VolatilityBucket"],
    "SMA200": ["SMA200Bucket"],

    "Market_x_Vol": [
        "MarketRegime",
        "VolatilityBucket",
    ],

    "Market_x_M20": [
        "MarketRegime",
        "Momentum20Bucket",
    ],

    "M20_x_Vol": [
        "Momentum20Bucket",
        "VolatilityBucket",
    ],

    "Full": regime_cols,
}

for year, yg in selected.groupby("Year"):

    for group_name, cols in GROUPINGS.items():

        for key, g in yg.groupby(
            cols,
            observed=True
        ):

            if len(g) < 50:
                continue

            if not isinstance(key, tuple):
                key = (key,)

            auc, bal, acc = metrics(g)

            diagnostic_rows.append({
                "Year": year,
                "Grouping": group_name,
                "Regime": " | ".join(map(str, key)),
                "Samples": len(g),
                "AUC": auc,
                "BalancedAccuracy": bal,
                "Accuracy": acc,
                "MeanProbability": g["probability"].mean(),
                "ActualUpRate": g["actual"].mean(),
            })

diag = pd.DataFrame(diagnostic_rows)

diag.to_csv(OUTPUT, index=False)

# ----------------------------------------------------------
# BAD YEARS
# ----------------------------------------------------------

print()
print("=" * 100)
print("2023 / 2026 — WORST REGIMES")
print("=" * 100)

bad = diag[
    (diag["Year"].isin([2023, 2026])) &
    (diag["Samples"] >= 100) &
    (diag["AUC"].notna())
].sort_values("AUC")

print(
    bad.head(25)[
        [
            "Year",
            "Grouping",
            "Regime",
            "Samples",
            "AUC",
            "BalancedAccuracy",
        ]
    ].to_string(
        index=False,
        formatters={
            "AUC": lambda x: f"{x:.4f}",
            "BalancedAccuracy":
                lambda x: f"%{x*100:.2f}",
        }
    )
)

# ----------------------------------------------------------
# GOOD YEARS
# ----------------------------------------------------------

print()
print("=" * 100)
print("2021 / 2022 / 2025 — BEST REGIMES")
print("=" * 100)

good = diag[
    (diag["Year"].isin([2021, 2022, 2025])) &
    (diag["Samples"] >= 100) &
    (diag["AUC"].notna())
].sort_values("AUC", ascending=False)

print(
    good.head(25)[
        [
            "Year",
            "Grouping",
            "Regime",
            "Samples",
            "AUC",
            "BalancedAccuracy",
        ]
    ].to_string(
        index=False,
        formatters={
            "AUC": lambda x: f"{x:.4f}",
            "BalancedAccuracy":
                lambda x: f"%{x*100:.2f}",
        }
    )
)

# ----------------------------------------------------------
# CROSS-YEAR STABILITY
# ----------------------------------------------------------

print()
print("=" * 100)
print("CROSS-YEAR REGIME STABILITY")
print("=" * 100)

stable_rows = []

for (grouping, regime), g in diag.groupby(
    ["Grouping", "Regime"]
):

    g = g[
        (g["Samples"] >= 100) &
        g["AUC"].notna()
    ]

    if len(g) < 4:
        continue

    stable_rows.append({
        "Grouping": grouping,
        "Regime": regime,
        "Years": len(g),
        "MeanAUC": g["AUC"].mean(),
        "MedianAUC": g["AUC"].median(),
        "MinAUC": g["AUC"].min(),
        "MaxAUC": g["AUC"].max(),
        "WinningYears": (g["AUC"] > .50).sum(),
        "WinRate": (g["AUC"] > .50).mean(),
        "Samples": g["Samples"].sum(),
    })

stable = pd.DataFrame(stable_rows)

stable = stable.sort_values(
    ["WinRate", "MeanAUC", "Samples"],
    ascending=False
)

print(
    stable.head(30).to_string(
        index=False,
        formatters={
            "MeanAUC": lambda x: f"{x:.4f}",
            "MedianAUC": lambda x: f"{x:.4f}",
            "MinAUC": lambda x: f"{x:.4f}",
            "MaxAUC": lambda x: f"{x:.4f}",
            "WinRate": lambda x: f"%{x*100:.1f}",
        }
    )
)

stable.to_csv(
    "data/gate_1260_053_cross_year_stability.csv",
    index=False
)

selected.to_csv(
    "data/gate_1260_053_selected_predictions.csv",
    index=False
)

print()
print("=" * 100)
print("KAYDEDİLDİ")
print("=" * 100)

print(OUTPUT)
print("data/gate_1260_053_cross_year_stability.csv")
print("data/gate_1260_053_selected_predictions.csv")

print()
print(
    f"Toplam süre: {(time.time()-t0)/60:.2f} dakika"
)
