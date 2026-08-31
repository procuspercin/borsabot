import inspect
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import src.test_base_30d_walkforward_regime_gate as base

DATA_DIR = ROOT / "data"
PRED_DIR = DATA_DIR / "volatility_individual_ablation_predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

OLD_SUMMARY = DATA_DIR / "base_30d_feature_ablation_summary.csv"
OLD_YEARLY = DATA_DIR / "base_30d_feature_ablation_yearly.csv"

OUT_SUMMARY = DATA_DIR / "base_30d_volatility_individual_ablation_summary.csv"
OUT_YEARLY = DATA_DIR / "base_30d_volatility_individual_ablation_yearly.csv"

TARGET_FEATURES = [
    "bollinger_width",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
]

PURE_VOLATILITY = [
    "atr_ratio",
    "bollinger_width",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
]

def safe_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)

def metrics(pred):
    y = pd.to_numeric(pred["actual"], errors="coerce")
    p = pd.to_numeric(pred["probability"], errors="coerce")
    ok = y.notna() & p.notna()
    y = y[ok].astype(int).to_numpy()
    p = p[ok].astype(float).to_numpy()
    yh = (p >= 0.50).astype(int)
    return {
        "Samples": len(y),
        "AUC": safe_auc(y, p),
        "BalancedAccuracy": balanced_accuracy_score(y, yh) if len(np.unique(y)) >= 2 else np.nan,
        "Accuracy": np.mean(yh == y),
    }

def normalize(result):
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if isinstance(result, (tuple, list)):
        for x in result:
            if isinstance(x, pd.DataFrame) and {"actual", "probability"}.issubset(x.columns):
                return x.copy()
    if isinstance(result, dict):
        for x in result.values():
            if isinstance(x, pd.DataFrame) and {"actual", "probability"}.issubset(x.columns):
                return x.copy()
    raise TypeError(f"run_test prediction DataFrame bulunamadı: {type(result)}")

def call_run_test(df, model_name, features):
    sig = inspect.signature(base.run_test)
    aliases = {
        "df": df, "data": df, "dataset": df,
        "model_name": model_name, "name": model_name,
        "numeric_features": features, "features": features,
        "feature_columns": features, "stock_features": features,
    }
    kwargs = {}
    missing = []
    for pname, param in sig.parameters.items():
        if pname in aliases:
            kwargs[pname] = aliases[pname]
        elif param.default is inspect._empty:
            missing.append(pname)
    if missing:
        raise RuntimeError(f"run_test zorunlu parametreleri çözülemedi: {missing}\n{sig}")
    return normalize(base.run_test(**kwargs))

def prepare_dataset():
    print("Dataset hazırlanıyor...")
    df = base.load_dataset()
    if hasattr(base, "add_market_regime"):
        df = base.add_market_regime(df)
    if hasattr(base, "add_target_end_date"):
        df = base.add_target_end_date(df)
    if hasattr(base, "add_gpr"):
        try:
            df = base.add_gpr(df)
        except Exception as e:
            print(f"Not: add_gpr atlandı: {e}")
    return df

def prediction_path(name):
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", name).lower()
    return PRED_DIR / f"{safe}.csv"

def yearly_metrics(pred, name):
    p = pred.copy()
    p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
    p["Year"] = p["Date"].dt.year
    rows = []
    for year, g in p.groupby("Year"):
        if pd.isna(year):
            continue
        m = metrics(g)
        rows.append({
            "Experiment": name,
            "Year": int(year),
            "Samples": m["Samples"],
            "AUC": m["AUC"],
            "BalancedAccuracy": m["BalancedAccuracy"],
            "Accuracy": m["Accuracy"],
        })
    return pd.DataFrame(rows)

def main():
    if not OLD_SUMMARY.exists() or not OLD_YEARLY.exists():
        raise FileNotFoundError("Önceki feature ablation CSV'leri bulunamadı.")

    old_s = pd.read_csv(OLD_SUMMARY)
    old_y = pd.read_csv(OLD_YEARLY)
    base_s = old_s[old_s["Experiment"] == "BASE_ALL"].iloc[0]
    base_y = old_y[old_y["Experiment"] == "BASE_ALL"].copy()

    base_auc = float(base_s["AUC"])
    base_bal = float(base_s["BalancedAccuracy"])

    print("=" * 110)
    print("BASE 30D — INDIVIDUAL VOLATILITY ABLATION")
    print("=" * 110)
    print(f"BASE yeniden eğitilmeyecek | AUC={base_auc:.4f} | BalAcc={base_bal*100:.2f}%")

    df = prepare_dataset()
    all_features = list(base.get_stock_features(df))
    print(f"Numeric feature sayısı: {len(all_features)}")

    experiments = []
    for f in TARGET_FEATURES:
        if f in all_features:
            experiments.append((f"NO_{f}", [x for x in all_features if x != f], [f]))

    # pure = [f for f in PURE_VOLATILITY if f in all_features]
    # experiments.append(("NO_PURE_VOLATILITY", [x for x in all_features if x not in set(pure)], pure))

    summary_rows = []
    yearly_all = []

    if OUT_SUMMARY.exists():
        summary_rows = pd.read_csv(OUT_SUMMARY).to_dict("records")
    if OUT_YEARLY.exists():
        yearly_all = [pd.read_csv(OUT_YEARLY)]

    completed = {r["Experiment"] for r in summary_rows}

    for i, (name, feats, removed) in enumerate(experiments, 1):
        print(f"\n>>> {i}/{len(experiments)} {name}")
        ppath = prediction_path(name)

        if name in completed and ppath.exists():
            print("Zaten tamamlanmış, atlanıyor.")
            continue

        if ppath.exists():
            print("Prediction CSV bulundu, yeniden eğitmeden kullanılıyor.")
            pred = pd.read_csv(ppath)
        else:
            t0 = time.time()
            pred = call_run_test(df, name, feats)
            pred.to_csv(ppath, index=False)
            print(f"\nSüre: {(time.time()-t0)/60:.2f} dk")

        m = metrics(pred)
        da = m["AUC"] - base_auc
        db = m["BalancedAccuracy"] - base_bal

        print(
            f"{name}: AUC={m['AUC']:.4f} ({da:+.4f}) | "
            f"BalAcc={m['BalancedAccuracy']*100:.2f}% ({db*100:+.2f} pp)"
        )

        summary_rows = [r for r in summary_rows if r.get("Experiment") != name]
        summary_rows.append({
            "Experiment": name,
            "RemovedFeatures": "|".join(removed),
            "RemovedCount": len(removed),
            "NumericFeatures": len(feats),
            "Samples": m["Samples"],
            "AUC": m["AUC"],
            "DeltaAUCvsBase": da,
            "BalancedAccuracy": m["BalancedAccuracy"],
            "DeltaBalAccVsBase": db,
            "Accuracy": m["Accuracy"],
        })

        ym = yearly_metrics(pred, name)
        if yearly_all:
            cur = pd.concat(yearly_all, ignore_index=True)
            cur = cur[cur["Experiment"] != name]
            yearly_all = [cur, ym]
        else:
            yearly_all = [ym]

        pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
        pd.concat(yearly_all, ignore_index=True).to_csv(OUT_YEARLY, index=False)

    summary = pd.read_csv(OUT_SUMMARY)
    yearly = pd.read_csv(OUT_YEARLY)

    by = base_y[["Year", "AUC", "BalancedAccuracy"]].rename(
        columns={"AUC": "BaseYearAUC", "BalancedAccuracy": "BaseYearBal"}
    )

    stability = []
    for exp, g in yearly.groupby("Experiment"):
        z = g.merge(by, on="Year", how="left")
        z["DeltaAUC"] = z["AUC"] - z["BaseYearAUC"]
        z["DeltaBal"] = z["BalancedAccuracy"] - z["BaseYearBal"]
        valid = z["DeltaAUC"].notna()
        stability.append({
            "Experiment": exp,
            "YearsBeatingBase": int((z.loc[valid, "DeltaAUC"] > 0).sum()),
            "ComparableYears": int(valid.sum()),
            "MeanYearDeltaAUC": z["DeltaAUC"].mean(),
            "MedianYearDeltaAUC": z["DeltaAUC"].median(),
            "MeanYearDeltaBalAcc": z["DeltaBal"].mean(),
        })

    final = summary.merge(pd.DataFrame(stability), on="Experiment", how="left")
    final = final.sort_values(["AUC", "MeanYearDeltaAUC"], ascending=False)

    print("\n" + "=" * 120)
    print("FINAL — INDIVIDUAL VOLATILITY ABLATION")
    print("=" * 120)

    show = final.copy()
    show["BalancedAccuracy"] *= 100
    show["DeltaBalAccVsBase"] *= 100
    show["MeanYearDeltaBalAcc"] *= 100

    print(show[
        ["Experiment", "AUC", "DeltaAUCvsBase", "BalancedAccuracy",
         "DeltaBalAccVsBase", "YearsBeatingBase", "ComparableYears",
         "MeanYearDeltaAUC", "MedianYearDeltaAUC", "MeanYearDeltaBalAcc"]
    ].to_string(index=False, formatters={
        "AUC": lambda x: f"{x:.4f}",
        "DeltaAUCvsBase": lambda x: f"{x:+.4f}",
        "BalancedAccuracy": lambda x: f"{x:.2f}%",
        "DeltaBalAccVsBase": lambda x: f"{x:+.2f} pp",
        "MeanYearDeltaAUC": lambda x: f"{x:+.4f}",
        "MedianYearDeltaAUC": lambda x: f"{x:+.4f}",
        "MeanYearDeltaBalAcc": lambda x: f"{x:+.2f} pp",
    }))

    print("\nKaydedildi:")
    print(f"  {OUT_SUMMARY}")
    print(f"  {OUT_YEARLY}")
    print(f"  {PRED_DIR}/")

if __name__ == "__main__":
    main()
