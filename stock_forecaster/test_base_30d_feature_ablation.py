"""
BASE 30D FEATURE ABLATION
=========================

Amaç:
- Mevcut src/test_base_30d_walkforward_regime_gate.py içindeki veri hazırlama,
  model ve walk-forward test mantığını aynen kullanmak.
- Sadece numeric feature setini değiştirerek hangi feature gruplarının
  BASE modele fayda/zarar verdiğini ölçmek.

Çalıştır:
    python test_base_30d_feature_ablation.py

Proje kök dizininden çalıştırılmalıdır.
"""

import inspect
import re
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import src.test_base_30d_walkforward_regime_gate as base


OUT_SUMMARY = Path("data/base_30d_feature_ablation_summary.csv")
OUT_YEARLY = Path("data/base_30d_feature_ablation_yearly.csv")
OUT_FEATURES = Path("data/base_30d_feature_ablation_feature_groups.csv")


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def safe_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def prediction_metrics(pred):
    if pred is None or len(pred) == 0:
        return {
            "Samples": 0,
            "Accuracy": np.nan,
            "BalancedAccuracy": np.nan,
            "AUC": np.nan,
        }

    required = {"actual", "probability"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(
            f"run_test sonucu beklenen sütunları içermiyor: {sorted(missing)}\n"
            f"Mevcut sütunlar: {list(pred.columns)}"
        )

    y = pd.to_numeric(pred["actual"], errors="coerce")
    p = pd.to_numeric(pred["probability"], errors="coerce")
    valid = y.notna() & p.notna()

    y = y.loc[valid].astype(int).to_numpy()
    p = p.loc[valid].astype(float).to_numpy()
    yhat = (p >= 0.50).astype(int)

    return {
        "Samples": len(y),
        "Accuracy": np.mean(yhat == y) if len(y) else np.nan,
        "BalancedAccuracy": (
            balanced_accuracy_score(y, yhat)
            if len(y) and len(np.unique(y)) >= 2
            else np.nan
        ),
        "AUC": safe_auc(y, p),
    }


def normalize_predictions(result):
    """
    run_test DataFrame döndürüyorsa direkt kullan.
    Tuple/list döndürüyorsa içindeki prediction DataFrame'ini bul.
    Dict döndürüyorsa uygun DataFrame değerini bul.
    """
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

    raise TypeError(
        "base.run_test() içinden prediction DataFrame otomatik bulunamadı.\n"
        f"Dönen tip: {type(result)}"
    )


def call_run_test(df, model_name, features):
    """
    Mevcut run_test fonksiyonunun parametre isimlerini inspect ederek çağırır.
    Böylece dosyadaki gerçek walk-forward mantığı aynen kullanılır.
    """
    sig = inspect.signature(base.run_test)
    kwargs = {}

    aliases = {
        "df": df,
        "data": df,
        "dataset": df,
        "model_name": model_name,
        "name": model_name,
        "numeric_features": features,
        "features": features,
        "feature_columns": features,
        "stock_features": features,
    }

    unresolved_required = []

    for pname, param in sig.parameters.items():
        if pname in aliases:
            kwargs[pname] = aliases[pname]
        elif param.default is inspect._empty:
            unresolved_required.append(pname)

    if unresolved_required:
        raise RuntimeError(
            "run_test() için otomatik çözülemeyen zorunlu parametre(ler): "
            + ", ".join(unresolved_required)
            + "\nSignature: "
            + str(sig)
        )

    result = base.run_test(**kwargs)
    return normalize_predictions(result)


def prepare_dataset():
    """
    main() içindeki hazırlama akışını mümkün olduğunca mevcut fonksiyonlarla
    yeniden kurar. Fonksiyon var ise çağrılır.
    """
    print("Dataset hazırlanıyor...")

    df = base.load_dataset()

    if hasattr(base, "add_market_regime"):
        df = base.add_market_regime(df)

    if hasattr(base, "add_target_end_date"):
        df = base.add_target_end_date(df)

    # BASE testte GPR stock feature olarak zaten dışlanıyor.
    # Yine de orijinal main() veri setine GPR ekliyorsa aynı dataframe yapısını
    # korumak için add_gpr'yi deniyoruz. Dosya/yapı uygun değilse BASE için
    # kritik olmadığı için devam ediyoruz.
    if hasattr(base, "add_gpr"):
        try:
            df = base.add_gpr(df)
        except Exception as exc:
            print(f"Not: add_gpr atlandı ({exc})")

    return df


# ---------------------------------------------------------------------
# FEATURE GROUPING
# ---------------------------------------------------------------------

def group_features(features):
    """
    İsim tabanlı, deterministik gruplama.
    Bir feature birden fazla ekonomik temaya uyabilse de ablation testinde
    her grup bağımsız olarak BASE'den çıkarılır.
    """

    patterns = {
        "RETURN_MOMENTUM": [
            r"return", r"ret_", r"_ret", r"momentum", r"\bmom\b",
            r"roc", r"change", r"pct", r"log_return"
        ],
        "RSI_OSCILLATORS": [
            r"rsi", r"stoch", r"williams", r"cci", r"mfi",
            r"ultimate", r"oscillator"
        ],
        "VOLATILITY_RANGE": [
            r"volatility", r"\bvol_", r"_vol\b", r"atr", r"true_range",
            r"range", r"boll", r"bb_", r"_bb", r"std", r"variance"
        ],
        "TREND_MA_DISTANCE": [
            r"sma", r"ema", r"ma_", r"_ma", r"distance", r"trend",
            r"adx", r"aroon", r"ichimoku"
        ],
        "VOLUME_FLOW": [
            r"volume", r"obv", r"vwap", r"cmf", r"money_flow",
            r"accum", r"distribution"
        ],
        "PRICE_POSITION": [
            r"position", r"drawdown", r"high_", r"_high", r"low_",
            r"_low", r"52w", r"breakout", r"support", r"resistance"
        ],
        "CALENDAR_TIME": [
            r"dayof", r"weekday", r"month", r"quarter", r"weekof",
            r"year_", r"sin_", r"cos_"
        ],
    }

    groups = {}

    for group_name, pats in patterns.items():
        matched = []
        for f in features:
            lf = str(f).lower()
            if any(re.search(p, lf) for p in pats):
                matched.append(f)
        if matched:
            groups[group_name] = sorted(set(matched))

    # İsim kalıplarına girmeyen feature'ları ayrıca göster.
    covered = set()
    for xs in groups.values():
        covered.update(xs)

    other = [f for f in features if f not in covered]
    if other:
        groups["OTHER"] = other

    return groups


def print_feature_groups(features, groups):
    print()
    print("=" * 112)
    print(f"BASE NUMERIC FEATURES ({len(features)})")
    print("=" * 112)
    for i, f in enumerate(features, 1):
        print(f"{i:>2}. {f}")

    print()
    print("=" * 112)
    print("AUTO FEATURE GROUPS")
    print("=" * 112)

    for group, cols in groups.items():
        print(f"\n[{group}] ({len(cols)})")
        for c in cols:
            print(f"  - {c}")


# ---------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------

def run_one(df, name, features):
    print()
    print("=" * 112)
    print(f"{name} | {len(features)} numeric feature")
    print("=" * 112)

    t0 = time.time()
    pred = call_run_test(df, name, features)

    if "Date" in pred.columns:
        pred["Date"] = pd.to_datetime(pred["Date"], errors="coerce")
        pred["Year"] = pred["Date"].dt.year

    m = prediction_metrics(pred)

    print()
    print(
        f"{name}: "
        f"Samples={m['Samples']} | "
        f"AUC={m['AUC']:.4f} | "
        f"BalAcc={m['BalancedAccuracy']*100:.2f}% | "
        f"Acc={m['Accuracy']*100:.2f}% | "
        f"Süre={(time.time()-t0)/60:.2f} dk"
    )

    return pred, m


def main():
    Path("data").mkdir(exist_ok=True)

    print("=" * 112)
    print("30D BASE — FEATURE GROUP ABLATION")
    print("=" * 112)
    print("Model/target/walk-forward mantığı mevcut BASE dosyasından alınır.")
    print("Bu testte yalnızca numeric feature seti değiştirilir.")

    df = prepare_dataset()

    all_features = list(base.get_stock_features(df))
    groups = group_features(all_features)

    print_feature_groups(all_features, groups)

    feature_rows = []
    for group, cols in groups.items():
        for c in cols:
            feature_rows.append({"Group": group, "Feature": c})
    pd.DataFrame(feature_rows).to_csv(OUT_FEATURES, index=False)

    experiments = [("BASE_ALL", all_features)]

    # Leave-one-GROUP-out
    for group, removed in groups.items():
        remaining = [f for f in all_features if f not in set(removed)]

        # Çok geniş OTHER grubu ayrı bir ekonomik grup değil; yine de raporlarız
        # fakat 2'den az feature kalacaksa testi yapmayız.
        if len(remaining) >= 2:
            experiments.append((f"NO_{group}", remaining))

    summary_rows = []
    yearly_rows = []

    base_auc = None
    base_bal = None

    for idx, (name, feats) in enumerate(experiments, 1):
        print()
        print(f"\n>>> EXPERIMENT {idx}/{len(experiments)}: {name}")

        pred, m = run_one(df, name, feats)

        if name == "BASE_ALL":
            base_auc = m["AUC"]
            base_bal = m["BalancedAccuracy"]

        summary_rows.append({
            "Experiment": name,
            "NumericFeatures": len(feats),
            "RemovedFeatures": len(all_features) - len(feats),
            "Samples": m["Samples"],
            "AUC": m["AUC"],
            "DeltaAUCvsBase": (
                m["AUC"] - base_auc
                if base_auc is not None and pd.notna(m["AUC"]) and pd.notna(base_auc)
                else np.nan
            ),
            "BalancedAccuracy": m["BalancedAccuracy"],
            "DeltaBalAccVsBase": (
                m["BalancedAccuracy"] - base_bal
                if base_bal is not None
                and pd.notna(m["BalancedAccuracy"])
                and pd.notna(base_bal)
                else np.nan
            ),
            "Accuracy": m["Accuracy"],
        })

        if "Year" in pred.columns:
            for year, g in pred.groupby("Year"):
                if pd.isna(year):
                    continue
                ym = prediction_metrics(g)
                yearly_rows.append({
                    "Experiment": name,
                    "Year": int(year),
                    "Samples": ym["Samples"],
                    "AUC": ym["AUC"],
                    "BalancedAccuracy": ym["BalancedAccuracy"],
                    "Accuracy": ym["Accuracy"],
                })

        # Her experiment sonrası ara kaydet.
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(OUT_SUMMARY, index=False)
        pd.DataFrame(yearly_rows).to_csv(OUT_YEARLY, index=False)

    summary = pd.DataFrame(summary_rows)

    # BASE'e göre sıralama.
    summary = summary.sort_values(
        ["DeltaAUCvsBase", "DeltaBalAccVsBase"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    summary.to_csv(OUT_SUMMARY, index=False)

    print()
    print("=" * 112)
    print("FINAL ABLATION SUMMARY")
    print("=" * 112)

    show = summary.copy()
    show["BalancedAccuracy"] *= 100
    show["DeltaBalAccVsBase"] *= 100
    show["Accuracy"] *= 100

    print(
        show.to_string(
            index=False,
            formatters={
                "AUC": lambda x: "NaN" if pd.isna(x) else f"{x:.4f}",
                "DeltaAUCvsBase": lambda x: "NaN" if pd.isna(x) else f"{x:+.4f}",
                "BalancedAccuracy": lambda x: "NaN" if pd.isna(x) else f"%{x:.2f}",
                "DeltaBalAccVsBase": lambda x: "NaN" if pd.isna(x) else f"{x:+.2f} pp",
                "Accuracy": lambda x: "NaN" if pd.isna(x) else f"%{x:.2f}",
            }
        )
    )

    print()
    print("Yorum:")
    print("  DeltaAUCvsBase > 0  => o grubu ÇIKARMAK modeli iyileştirmiş olabilir.")
    print("  DeltaAUCvsBase < 0  => o grup modele fayda sağlıyor olabilir.")
    print("  Tek bir backtest sonucuyla feature silmeyeceğiz; iyi adayları sonraki aşamada")
    print("  yıl bazında ve daha dar ablation ile doğrulayacağız.")

    print()
    print("Kaydedildi:")
    print(f"  {OUT_SUMMARY}")
    print(f"  {OUT_YEARLY}")
    print(f"  {OUT_FEATURES}")


if __name__ == "__main__":
    main()
