import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

INPUT = Path("data/base_30d_regime_predictions_enriched.csv")

WINDOW = 1260
TARGET_HORIZON = 30
MIN_HISTORY = 300

LOWER_AUC = 0.53
UPPER_AUC = 0.60

LEVELS = [
    ("FULL", [
        "MarketRegime",
        "Momentum20Bucket",
        "Momentum60Bucket",
        "VolatilityBucketRolling",
        "SMA200Bucket",
    ]),
    ("TREND_VOL_M20", [
        "MarketRegime",
        "VolatilityBucketRolling",
        "Momentum20Bucket",
    ]),
    ("TREND_VOL", [
        "MarketRegime",
        "VolatilityBucketRolling",
    ]),
    ("M20_VOL", [
        "Momentum20Bucket",
        "VolatilityBucketRolling",
    ]),
    ("M20", ["Momentum20Bucket"]),
    ("MARKET", ["MarketRegime"]),
]


def safe_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def metrics(g):
    if len(g) == 0:
        return {
            "Samples": 0,
            "Accuracy": np.nan,
            "BalancedAccuracy": np.nan,
            "AUC": np.nan,
        }

    y = g["actual"].to_numpy()
    p = g["probability"].to_numpy()
    pred = (p >= 0.50).astype(int)

    auc = safe_auc(y, p)
    bal = (
        balanced_accuracy_score(y, pred)
        if len(np.unique(y)) >= 2
        else np.nan
    )

    return {
        "Samples": len(g),
        "Accuracy": np.mean(pred == y),
        "BalancedAccuracy": bal,
        "AUC": auc,
    }


def group_map(hist, cols):
    result = {}

    for key, g in hist.groupby(cols, observed=True, sort=False):
        if len(g) < MIN_HISTORY:
            continue

        if not isinstance(key, tuple):
            key = (key,)

        y = g["actual"].to_numpy()
        p = g["probability"].to_numpy()
        pred = (p >= 0.50).astype(int)

        auc = safe_auc(y, p)
        if pd.isna(auc):
            continue

        bal = (
            balanced_accuracy_score(y, pred)
            if len(np.unique(y)) >= 2
            else np.nan
        )

        result[tuple(str(x) for x in key)] = {
            "samples": len(g),
            "auc": auc,
            "bal": bal,
        }

    return result


def main():
    print("=" * 112)
    print("30D BASE — PRODUCTION-SAFE ROLLING VOLATILITY BUCKET + 0.53–0.60 GATE")
    print("=" * 112)
    print(f"Window          : {WINDOW} trading dates")
    print(f"Target horizon  : {TARGET_HORIZON} trading dates")
    print(f"Min history     : {MIN_HISTORY}")
    print(f"Gate            : {LOWER_AUC:.2f} <= historical AUC < {UPPER_AUC:.2f}")
    print("Volatility bins : rolling q33/q67 from PRIOR market dates only")
    print()

    t0 = time.time()

    df = pd.read_csv(INPUT, parse_dates=["Date"], low_memory=False)
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    required = [
        "Date", "Ticker", "actual", "probability",
        "MarketRegime", "Momentum20Bucket", "Momentum60Bucket",
        "market_volatility_20d", "SMA200Bucket"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for c in ["MarketRegime", "Momentum20Bucket", "Momentum60Bucket", "SMA200Bucket"]:
        df[c] = df[c].astype("string").fillna("NA")

    dates = df["Date"].drop_duplicates().sort_values().reset_index(drop=True)
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}

    df["_date_idx"] = df["Date"].map(date_to_idx).astype(int)
    df["_label_available_idx"] = df["_date_idx"] + TARGET_HORIZON
    df["Year"] = df["Date"].dt.year

    # One market volatility value per trading date.
    # Median is used defensively in case repeated ticker rows differ slightly.
    market_daily = (
        df.groupby("_date_idx", as_index=False)["market_volatility_20d"]
        .median()
        .sort_values("_date_idx")
        .reset_index(drop=True)
    )
    vol_by_idx = dict(
        zip(
            market_daily["_date_idx"].astype(int),
            market_daily["market_volatility_20d"].astype(float)
        )
    )

    by_date = {
        int(di): g.index.to_numpy()
        for di, g in df.groupby("_date_idx", sort=True)
    }

    decision_records = []
    vol_cut_records = []

    for di in range(len(dates)):
        today_ids = by_date.get(di)
        if today_ids is None:
            continue

        # ------------------------------------------------------
        # ROLLING VOLATILITY CUTS: prior dates only
        # ------------------------------------------------------
        vol_start = max(0, di - WINDOW)

        prior_vol = market_daily[
            (market_daily["_date_idx"] >= vol_start)
            & (market_daily["_date_idx"] < di)
        ]["market_volatility_20d"].dropna()

        if len(prior_vol) >= 60:
            q33 = prior_vol.quantile(1/3)
            q67 = prior_vol.quantile(2/3)
        else:
            q33 = np.nan
            q67 = np.nan

        vol_cut_records.append({
            "_date_idx": di,
            "Date": dates.iloc[di],
            "VolHistoryDates": len(prior_vol),
            "Q33": q33,
            "Q67": q67,
        })

        # We need rolling buckets for historical rows using TODAY's
        # past-only thresholds. This keeps regime matching leakage-safe.
        hist_start = max(0, di - WINDOW)

        hist = df[
            (df["_label_available_idx"] <= di)
            & (df["_date_idx"] >= hist_start)
        ].copy()

        today = df.loc[today_ids].copy()

        if pd.notna(q33) and pd.notna(q67):
            def bucket_series(s):
                return pd.cut(
                    s,
                    bins=[-np.inf, q33, q67, np.inf],
                    labels=["LOW", "MEDIUM", "HIGH"],
                    include_lowest=True,
                    right=True,
                ).astype("string")

            hist["VolatilityBucketRolling"] = bucket_series(
                hist["market_volatility_20d"]
            ).fillna("NA")

            today["VolatilityBucketRolling"] = bucket_series(
                today["market_volatility_20d"]
            ).fillna("NA")
        else:
            hist["VolatilityBucketRolling"] = "NA"
            today["VolatilityBucketRolling"] = "NA"

        maps = {}

        if len(hist) >= MIN_HISTORY:
            for level_name, cols in LEVELS:
                maps[level_name] = (cols, group_map(hist, cols))

        for rid, row in today.iterrows():
            chosen_level = "NONE"
            hist_samples = 0
            hist_auc = np.nan
            hist_bal = np.nan

            for level_name, cols in LEVELS:
                if level_name not in maps:
                    continue

                _, gmap = maps[level_name]
                key = tuple(str(row[c]) for c in cols)

                if key in gmap:
                    m = gmap[key]
                    chosen_level = level_name
                    hist_samples = m["samples"]
                    hist_auc = m["auc"]
                    hist_bal = m["bal"]
                    break

            active = (
                pd.notna(hist_auc)
                and LOWER_AUC <= hist_auc < UPPER_AUC
                and pd.notna(hist_bal)
                and hist_bal >= 0.50
            )

            decision_records.append({
                "_row_id": rid,
                "GateActive": active,
                "GateLevel": chosen_level,
                "GateHistSamples": hist_samples,
                "GateHistAUC": hist_auc,
                "GateHistBalAcc": hist_bal,
                "VolatilityBucketRolling": row["VolatilityBucketRolling"],
                "RollingVolQ33": q33,
                "RollingVolQ67": q67,
            })

        if di % 300 == 0 or di == len(dates) - 1:
            print(f"{di + 1}/{len(dates)} | {pd.Timestamp(dates.iloc[di]).date()}")

    decisions = pd.DataFrame(decision_records).set_index("_row_id").sort_index()
    out = df.join(decisions)

    # ------------------------------------------------------
    # OVERALL
    # ------------------------------------------------------
    base_m = metrics(out)
    gate = out[out["GateActive"] == True].copy()
    gate_m = metrics(gate)

    summary = pd.DataFrame([
        {
            "Strategy": "BASE_ALL",
            **base_m,
            "Coverage": 1.0,
        },
        {
            "Strategy": "ROLLING_VOL_GATE_053_060",
            **gate_m,
            "Coverage": len(gate) / len(out),
        },
    ])

    print()
    print("=" * 112)
    print("OVERALL")
    print("=" * 112)

    printable = summary.copy()
    printable["Coverage"] *= 100
    printable["Accuracy"] *= 100
    printable["BalancedAccuracy"] *= 100

    print(
        printable.to_string(
            index=False,
            formatters={
                "Coverage": lambda x: f"%{x:.2f}",
                "Accuracy": lambda x: f"%{x:.2f}",
                "BalancedAccuracy": lambda x: f"%{x:.2f}",
                "AUC": lambda x: "NaN" if pd.isna(x) else f"{x:.4f}",
            }
        )
    )

    # ------------------------------------------------------
    # YEARLY
    # ------------------------------------------------------
    yearly_rows = []

    for year in sorted(out["Year"].dropna().unique()):
        all_y = out[out["Year"] == year]
        gate_y = all_y[all_y["GateActive"] == True]

        bm = metrics(all_y)
        gm = metrics(gate_y)

        yearly_rows.append({
            "Year": int(year),
            "Samples": gm["Samples"],
            "Coverage": len(gate_y) / len(all_y) if len(all_y) else np.nan,
            "BaseAUC": bm["AUC"],
            "GateAUC": gm["AUC"],
            "DeltaVsBase": (
                gm["AUC"] - bm["AUC"]
                if pd.notna(gm["AUC"]) and pd.notna(bm["AUC"])
                else np.nan
            ),
            "BalancedAccuracy": gm["BalancedAccuracy"],
            "Accuracy": gm["Accuracy"],
        })

    yearly = pd.DataFrame(yearly_rows)

    print()
    print("=" * 112)
    print("YEAR BY YEAR")
    print("=" * 112)

    py = yearly.copy()
    py["Coverage"] *= 100
    py["BalancedAccuracy"] *= 100
    py["Accuracy"] *= 100

    print(
        py.to_string(
            index=False,
            formatters={
                "Coverage": lambda x: f"%{x:.2f}",
                "BaseAUC": lambda x: "NaN" if pd.isna(x) else f"{x:.4f}",
                "GateAUC": lambda x: "NaN" if pd.isna(x) else f"{x:.4f}",
                "DeltaVsBase": lambda x: "NaN" if pd.isna(x) else f"{x:+.4f}",
                "BalancedAccuracy": lambda x: "NaN" if pd.isna(x) else f"%{x:.2f}",
                "Accuracy": lambda x: "NaN" if pd.isna(x) else f"%{x:.2f}",
            }
        )
    )

    comparable = yearly.dropna(subset=["GateAUC", "BaseAUC"])
    wins = int((comparable["GateAUC"] > comparable["BaseAUC"]).sum())

    print()
    print(
        f"BASE'i geçen yıl: {wins}/{len(comparable)} "
        f"(%{100*wins/len(comparable):.1f})"
        if len(comparable)
        else "Karşılaştırılabilir yıl yok."
    )

    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------
    Path("data").mkdir(exist_ok=True)

    summary.to_csv(
        "data/base_30d_rolling_vol_gate_053_060_summary.csv",
        index=False
    )
    yearly.to_csv(
        "data/base_30d_rolling_vol_gate_053_060_yearly.csv",
        index=False
    )
    out.to_csv(
        "data/base_30d_rolling_vol_gate_053_060_predictions.csv",
        index=False
    )
    pd.DataFrame(vol_cut_records).to_csv(
        "data/base_30d_rolling_vol_cutoffs.csv",
        index=False
    )

    print()
    print("=" * 112)
    print("SAVED")
    print("=" * 112)
    print("data/base_30d_rolling_vol_gate_053_060_summary.csv")
    print("data/base_30d_rolling_vol_gate_053_060_yearly.csv")
    print("data/base_30d_rolling_vol_gate_053_060_predictions.csv")
    print("data/base_30d_rolling_vol_cutoffs.csv")
    print()
    print(f"Toplam süre: {(time.time() - t0)/60:.2f} dakika")


if __name__ == "__main__":
    main()
