from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

PRED_PATH = Path("data/base_30d_regime_predictions_enriched.csv")
WINDOWS = [252, 504, 756, 1260]
AUC_THRESHOLDS = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]
MIN_HISTORY = 300
TARGET_HORIZON = 30

REGIME_LEVELS = [
    ("FULL", ["MarketRegime","Momentum20Bucket","Momentum60Bucket","VolatilityBucket","SMA200Bucket"]),
    ("TREND_VOL_M20", ["MarketRegime","VolatilityBucket","Momentum20Bucket"]),
    ("TREND_VOL", ["MarketRegime","VolatilityBucket"]),
    ("M20_VOL", ["Momentum20Bucket","VolatilityBucket"]),
    ("M20", ["Momentum20Bucket"]),
    ("MARKET", ["MarketRegime"]),
]

def safe_auc(y_true, probability):
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    return np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, probability)

def evaluate_subset(df):
    if df.empty:
        return {"Samples":0,"Accuracy":np.nan,"BalancedAccuracy":np.nan,"AUC":np.nan}
    pred = (df["probability"].to_numpy() >= 0.50).astype(int)
    return {
        "Samples": len(df),
        "Accuracy": (pred == df["actual"].to_numpy()).mean(),
        "BalancedAccuracy": balanced_accuracy_score(df["actual"].to_numpy(), pred),
        "AUC": safe_auc(df["actual"].to_numpy(), df["probability"].to_numpy()),
    }

def load_data():
    print("Prediction/regime verisi yükleniyor...")
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"{PRED_PATH} bulunamadı. Önce regime analysis scriptini çalıştır.")
    df = pd.read_csv(PRED_PATH, parse_dates=["Date"], low_memory=False)
    needed = ["Date","Ticker","actual","probability","MarketRegime","Momentum20Bucket","Momentum60Bucket","VolatilityBucket","SMA200Bucket"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")
    df = df.sort_values(["Date","Ticker"]).reset_index(drop=True)
    for c in ["MarketRegime","Momentum20Bucket","Momentum60Bucket","VolatilityBucket","SMA200Bucket"]:
        df[c] = df[c].astype("string").fillna("NA")
    dates = df["Date"].drop_duplicates().sort_values().reset_index(drop=True)
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    df["_date_idx"] = df["Date"].map(date_to_idx)
    df["_label_available_idx"] = df["_date_idx"] + TARGET_HORIZON
    df["Year"] = df["Date"].dt.year
    return df, dates

def build_gate_metrics(df, dates, window):
    print(f"Window {window}: rolling history...")
    by_date = {int(di): grp.index.to_numpy() for di, grp in df.groupby("_date_idx", sort=True)}
    rows = []
    total = len(dates)
    for di in range(total):
        today_idx = by_date.get(di)
        if today_idx is None:
            continue
        hist_start = max(0, di - window)
        hist = df[(df["_label_available_idx"] <= di) & (df["_date_idx"] >= hist_start)]
        if len(hist) < MIN_HISTORY:
            for rid in today_idx:
                rows.append({"_row_id":rid,"GateLevel":"NONE","GateHistorySamples":0,"GateHistoricalAUC":np.nan,"GateHistoricalBalancedAccuracy":np.nan})
            continue
        level_maps = {}
        for level_name, cols in REGIME_LEVELS:
            gmap = {}
            for key, g in hist.groupby(cols, observed=True, sort=False):
                if len(g) < MIN_HISTORY:
                    continue
                if not isinstance(key, tuple):
                    key = (key,)
                pred = (g["probability"].to_numpy() >= 0.50).astype(int)
                gmap[tuple(str(x) for x in key)] = (
                    len(g),
                    safe_auc(g["actual"].to_numpy(), g["probability"].to_numpy()),
                    balanced_accuracy_score(g["actual"].to_numpy(), pred),
                )
            level_maps[level_name] = (cols, gmap)
        today = df.loc[today_idx]
        for rid, row in today.iterrows():
            chosen_level, n_hist, auc_hist, bal_hist = "NONE", 0, np.nan, np.nan
            for level_name, (cols, gmap) in level_maps.items():
                key = tuple(str(row[c]) for c in cols)
                if key in gmap:
                    chosen_level = level_name
                    n_hist, auc_hist, bal_hist = gmap[key]
                    break
            rows.append({"_row_id":rid,"GateLevel":chosen_level,"GateHistorySamples":n_hist,"GateHistoricalAUC":auc_hist,"GateHistoricalBalancedAccuracy":bal_hist})
        if di % 250 == 0 or di == total - 1:
            print(f"  {di+1}/{total} | {pd.Timestamp(dates.iloc[di]).date()}")
    return pd.DataFrame(rows).set_index("_row_id").sort_index()

def run_grid(df, dates):
    base_metrics = evaluate_subset(df)
    rows, yearly_rows = [], []
    for window in WINDOWS:
        t0 = time.time()
        decisions = build_gate_metrics(df, dates, window)
        work = df.join(decisions, how="left")
        for threshold in AUC_THRESHOLDS:
            active = (
                work["GateHistoricalAUC"].notna()
                & (work["GateHistoricalAUC"] >= threshold)
                & (work["GateHistoricalBalancedAccuracy"] >= 0.50)
            )
            gated = work.loc[active].copy()
            gm = evaluate_subset(gated)
            year_wins = 0
            valid_years = 0
            for year in sorted(work["Year"].dropna().unique()):
                y_all = work[work["Year"] == year]
                y_gate = gated[gated["Year"] == year]
                base_y = evaluate_subset(y_all)
                gate_y = evaluate_subset(y_gate)
                if pd.notna(base_y["AUC"]) and pd.notna(gate_y["AUC"]) and gate_y["Samples"] >= 100:
                    valid_years += 1
                    if gate_y["AUC"] > base_y["AUC"]:
                        year_wins += 1
                yearly_rows.append({
                    "Window":window,"Threshold":threshold,"Year":int(year),
                    "GateSamples":gate_y["Samples"],
                    "Coverage":gate_y["Samples"]/len(y_all) if len(y_all) else np.nan,
                    "GateAUC":gate_y["AUC"],"BaseAUC":base_y["AUC"],
                    "DeltaAUC":gate_y["AUC"]-base_y["AUC"] if pd.notna(gate_y["AUC"]) and pd.notna(base_y["AUC"]) else np.nan,
                    "GateBalancedAccuracy":gate_y["BalancedAccuracy"],
                    "BaseBalancedAccuracy":base_y["BalancedAccuracy"],
                })
            rows.append({
                "Window":window,"Threshold":threshold,"Samples":gm["Samples"],
                "Coverage":gm["Samples"]/len(df) if len(df) else np.nan,
                "Accuracy":gm["Accuracy"],"BalancedAccuracy":gm["BalancedAccuracy"],"AUC":gm["AUC"],
                "DeltaAUC_vs_BASE":gm["AUC"]-base_metrics["AUC"] if pd.notna(gm["AUC"]) else np.nan,
                "DeltaBalAcc_vs_BASE":gm["BalancedAccuracy"]-base_metrics["BalancedAccuracy"] if pd.notna(gm["BalancedAccuracy"]) else np.nan,
                "YearsBeatingBASE":year_wins,"ValidYears":valid_years,
                "YearWinRate":year_wins/valid_years if valid_years else np.nan,
            })
        print(f"Window {window} tamamlandı ({time.time()-t0:.1f} sn)")
    return pd.DataFrame(rows), pd.DataFrame(yearly_rows), base_metrics

def main():
    print("="*110)
    print("30D BASE — ROLLING REGIME GATE GRID SEARCH — FAST VERSION")
    print("="*110)
    print("Windows:", WINDOWS)
    print("Thresholds:", AUC_THRESHOLDS)
    print("Min hist:", MIN_HISTORY)
    t0 = time.time()
    df, dates = load_data()
    print("Rows:", f"{len(df):,}")
    print("Dates:", len(dates))
    summary, yearly, base = run_grid(df, dates)
    summary = summary.sort_values(["YearsBeatingBASE","AUC","BalancedAccuracy","Coverage"], ascending=[False,False,False,False]).reset_index(drop=True)
    print("\nBASE")
    print(f"Samples={base['Samples']:,} | AUC={base['AUC']:.4f} | BalAcc=%{100*base['BalancedAccuracy']:.2f}")
    d = summary.copy()
    for c in ["Coverage","Accuracy","BalancedAccuracy","YearWinRate"]:
        d[c] *= 100
    print("\nGRID SEARCH SONUÇLARI\n")
    print(d.to_string(index=False, formatters={
        "Threshold": lambda x:f"{x:.2f}",
        "Coverage": lambda x:f"%{x:.2f}",
        "Accuracy": lambda x:f"%{x:.2f}",
        "BalancedAccuracy": lambda x:f"%{x:.2f}",
        "AUC": lambda x:f"{x:.4f}",
        "DeltaAUC_vs_BASE": lambda x:f"{x:+.4f}",
        "DeltaBalAcc_vs_BASE": lambda x:f"{100*x:+.2f} puan",
        "YearWinRate": lambda x:f"%{x:.1f}",
    }))
    print("\nTOP 10\n")
    print(d.head(10)[["Window","Threshold","Coverage","AUC","BalancedAccuracy","DeltaAUC_vs_BASE","YearsBeatingBASE","ValidYears","YearWinRate"]].to_string(index=False))
    Path("data").mkdir(exist_ok=True)
    summary.to_csv("data/base_30d_rolling_regime_gate_grid.csv", index=False)
    yearly.to_csv("data/base_30d_rolling_regime_gate_grid_yearly.csv", index=False)
    print(f"\nToplam süre: {(time.time()-t0)/60:.2f} dakika")

if __name__ == "__main__":
    main()
