from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from features import create_features
from market_features import create_market_features


RAW_DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")

MARKET_PATH = RAW_DATA_DIR / "XU100.csv"
CALIBRATION_PATH = MODEL_DIR / "expected_return_calibration.csv"

HORIZONS = [10, 30, 60, 120, 180]

MARKET_FEATURES = [
    "market_return_20d",
    "market_return_60d",
    "market_sma_50_distance",
    "market_sma_200_distance",
    "market_volatility_20d",
]


def calculate_market_regime(row):
    bullish = (
        row["market_return_20d"] > 0
        and row["market_return_60d"] > 0
        and row["market_sma_50_distance"] > 0
        and row["market_sma_200_distance"] > 0
    )

    bearish = (
        row["market_return_20d"] < 0
        and row["market_return_60d"] < 0
        and row["market_sma_50_distance"] < 0
        and row["market_sma_200_distance"] < 0
    )

    if bullish:
        return "BULL"

    if bearish:
        return "BEAR"

    return "NEUTRAL"


def load_stock(ticker):
    path = RAW_DATA_DIR / f"{ticker}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{ticker} verisi bulunamadı: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=["Date"],
        index_col="Date",
    )

    df = df.sort_index()

    # Yahoo, seans kapanmadan Close'u boş bir satır döndürebiliyor
    # (Open/High/Low/Volume dolu, Close boş). Bu satır sonuncu olduğunda
    # latest_close NaN oluyor ve bütün fiyat beklentileri hesaplanamıyor:
    # yüzdeler geliyor ama fiyatlar arayüzde "-" görünüyor. Kapanışı
    # olmayan satırları hiç işleme alma.
    return df[df["Close"].notna()]


def load_market():
    if not MARKET_PATH.exists():
        raise FileNotFoundError(
            "XU100 verisi bulunamadı."
        )

    df = pd.read_csv(
        MARKET_PATH,
        parse_dates=["Date"],
        index_col="Date",
    )

    df = df.sort_index()

    # Endeks verisinde de yarım satır gelebiliyor; aynı nedenle ayıkla.
    return df[df["Close"].notna()]


def load_calibration():
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"Calibration bulunamadı: {CALIBRATION_PATH}"
        )

    df = pd.read_csv(
        CALIBRATION_PATH
    )

    numeric_cols = [
        "Horizon",
        "BinLower",
        "BinUpper",
        "Samples",
        "MeanProbability",
        "ActualUpRate",
        "MeanReturn",
        "MedianReturn",
        "P25Return",
        "P75Return",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


def build_latest_row(ticker):
    stock_df = load_stock(ticker)
    market_df = load_market()

    stock_features = (
        create_features(stock_df)
        .reset_index()
    )

    market_features = (
        create_market_features(market_df)
        .reset_index()
    )

    merged = stock_features.merge(
        market_features,
        on="Date",
        how="left",
    )

    merged["Ticker"] = ticker

    merged["MarketRegime"] = merged.apply(
        calculate_market_regime,
        axis=1,
    )

    merged = merged.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    required = (
        MARKET_FEATURES
        + [
            "rsi_14",
            "Ticker",
            "MarketRegime",
        ]
    )

    latest_valid = merged.dropna(
        subset=required
    )

    if latest_valid.empty:
        raise RuntimeError(
            "Geçerli son feature satırı bulunamadı."
        )

    latest = (
        latest_valid
        .iloc[[-1]]
        .copy()
    )

    date = latest["Date"].iloc[0]

    close = float(
        stock_df.loc[date, "Close"]
        if date in stock_df.index
        else stock_df["Close"].iloc[-1]
    )

    return latest, date, close


def predict_horizon(
    row,
    horizon,
):
    model_path = (
        MODEL_DIR
        / f"direction_{horizon}d.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {model_path}"
        )

    model = joblib.load(
        model_path
    )

    probability = (
        model.predict_proba(row)[0, 1]
    )

    return float(probability)


def find_calibration(
    calibration,
    horizon,
    probability,
):
    subset = calibration[
        calibration["Horizon"] == horizon
    ].copy()

    if subset.empty:
        return None

    exact = subset[
        (probability >= subset["BinLower"])
        & (probability < subset["BinUpper"])
    ]

    if not exact.empty:
        return exact.iloc[0]

    subset["distance"] = abs(
        subset["MeanProbability"]
        - probability
    )

    return (
        subset
        .sort_values("distance")
        .iloc[0]
    )


def pct(value):
    return f"{value * 100:+.2f}%"


def tl(value):
    return (
        f"{value:,.2f} TL"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def describe_market(row):
    print()
    print("=" * 78)
    print("GENEL PİYASA")
    print("=" * 78)

    r20 = float(
        row["market_return_20d"]
    )

    r60 = float(
        row["market_return_60d"]
    )

    sma50 = float(
        row["market_sma_50_distance"]
    )

    sma200 = float(
        row["market_sma_200_distance"]
    )

    print()
    print(
        f"BIST 100 — son 20 işlem günü : {pct(r20)}"
    )

    print(
        "→ Genel piyasa son haftalarda "
        + (
            "yükselmiş."
            if r20 > 0
            else "gerilemiş."
        )
    )

    print()
    print(
        f"BIST 100 — son 60 işlem günü : {pct(r60)}"
    )

    print(
        "→ Daha uzun dönem piyasa hareketi "
        + (
            "yukarı yönlü."
            if r60 > 0
            else "aşağı yönlü."
        )
    )

    print()
    print(
        f"50 günlük ortalamaya uzaklık : {pct(sma50)}"
    )

    print(
        "→ Endeks son dönem ortalama seviyesinin "
        + (
            "üzerinde."
            if sma50 > 0
            else "altında."
        )
    )

    print()
    print(
        f"200 günlük ortalamaya uzaklık: {pct(sma200)}"
    )

    print(
        "→ Genel piyasa uzun dönem ortalamasının "
        + (
            "üzerinde."
            if sma200 > 0
            else "altında."
        )
    )


def describe_rsi(row):
    value = float(
        row["rsi_14"]
    )

    print(
        f"RSI (14)                 : {value:.1f}"
    )

    if value >= 70:
        print(
            "→ RSI 70'in üzerinde. Hisse son dönemde güçlü yükselmiş; "
            "aynı zamanda kısa vadede yorulma veya geri çekilme "
            "riski artmış olabilir."
        )

    elif value <= 30:
        print(
            "→ RSI 30'un altında. Hisse son dönemde yoğun satış baskısı "
            "yaşamış; tepki yükselişi ihtimali artmış olabilir."
        )

    elif value >= 55:
        print(
            "→ RSI orta bölgenin üzerinde. "
            "Son fiyat hareketlerinde alıcılar bir miktar daha güçlü."
        )

    elif value <= 45:
        print(
            "→ RSI orta bölgenin altında. "
            "Son fiyat hareketlerinde satış baskısı biraz daha güçlü."
        )

    else:
        print(
            "→ RSI dengeli bölgede. "
            "Bu gösterge tek başına belirgin bir yön göstermiyor."
        )


def describe_macd(row):
    if (
        "macd" not in row.index
        or "macd_signal" not in row.index
    ):
        return

    macd = float(row["macd"])
    signal = float(
        row["macd_signal"]
    )

    print()
    print(
        f"MACD                     : {macd:.3f}"
    )

    if macd > signal:
        print(
            "→ MACD kendi sinyal çizgisinin üzerinde. "
            "Son dönem fiyat hareketinde yukarı yönlü ivme destekleniyor."
        )

    elif macd < signal:
        print(
            "→ MACD kendi sinyal çizgisinin altında. "
            "Son dönem fiyat hareketinde güç kaybı görülüyor."
        )

    else:
        print(
            "→ MACD ile sinyal çizgisi birbirine yakın. "
            "Momentum tarafında belirgin üstünlük yok."
        )


def describe_stochastic(row):
    if "stoch_k" not in row.index:
        return

    value = float(
        row["stoch_k"]
    )

    print()
    print(
        f"Stochastic               : {value:.1f}"
    )

    if value >= 80:
        print(
            "→ Fiyat son dönem işlem aralığının üst bölümünde. "
            "Momentum güçlü; fakat kısa vadede fazla yükselmiş olma "
            "ihtimali de artıyor."
        )

    elif value <= 20:
        print(
            "→ Fiyat son dönem işlem aralığının alt bölümünde. "
            "Satış baskısı güçlü; tepki hareketi ihtimali oluşabilir."
        )

    else:
        print(
            "→ Fiyat son dönem işlem aralığının orta bölümünde."
        )


def describe_stock_trend(row):
    if "return_20d" in row.index:
        r20 = float(
            row["return_20d"]
        )

        print()
        print(
            f"Son 20 işlem günü        : {pct(r20)}"
        )

        print(
            "→ Hisse son haftalarda "
            + (
                "yükselmiş."
                if r20 > 0
                else "gerilemiş."
            )
        )

    if "return_60d" in row.index:
        r60 = float(
            row["return_60d"]
        )

        print()
        print(
            f"Son 60 işlem günü        : {pct(r60)}"
        )

        print(
            "→ Orta vadeli fiyat hareketi "
            + (
                "yukarı yönlü."
                if r60 > 0
                else "aşağı yönlü."
            )
        )


def describe_moving_averages(row):
    if "close_sma_50_ratio" in row.index:
        distance = float(
            row["close_sma_50_ratio"]
        )

        print()
        print(
            f"50 günlük ortalamaya uzaklık : {pct(distance)}"
        )

        print(
            "→ Güncel fiyat son 50 günlük ortalama fiyatının "
            + (
                "üzerinde."
                if distance > 0
                else "altında."
            )
        )

    if "close_sma_200_ratio" in row.index:
        distance = float(
            row["close_sma_200_ratio"]
        )

        print()
        print(
            f"200 günlük ortalamaya uzaklık: {pct(distance)}"
        )

        print(
            "→ Hisse uzun dönem ortalama fiyatının "
            + (
                "üzerinde seyrediyor."
                if distance > 0
                else "altında seyrediyor."
            )
        )


def describe_volume(row):
    if "volume_ratio_20" not in row.index:
        return

    difference = (
        float(
            row["volume_ratio_20"]
        )
        - 1
    )

    print()
    print(
        f"İşlem hacmi              : "
        f"20 günlük ortalamaya göre {pct(difference)}"
    )

    if difference > 0.20:
        print(
            "→ Son işlemlerde normalden belirgin şekilde "
            "daha yüksek yatırımcı aktivitesi var."
        )

    elif difference < -0.20:
        print(
            "→ Son işlem hacmi son dönem ortalamasının "
            "belirgin şekilde altında."
        )

    else:
        print(
            "→ İşlem hacmi son dönem normal seviyelerine yakın."
        )


def describe_volatility(row):
    if "volatility_20d" not in row.index:
        return

    value = float(
        row["volatility_20d"]
    )

    print()
    print(
        f"20 günlük oynaklık       : %{value * 100:.2f}"
    )

    if value >= 0.04:
        print(
            "→ Fiyat hareketleri son dönemde oldukça sert."
        )

    elif value >= 0.02:
        print(
            "→ Hisse son dönemde orta seviyede dalgalanıyor."
        )

    else:
        print(
            "→ Son dönemde fiyat hareketleri görece sakin."
        )


def describe_stock(row, ticker):
    print()
    print("=" * 78)
    print(
        f"{ticker} — TEKNİK GÖRÜNÜM"
    )
    print("=" * 78)
    print()

    describe_rsi(row)
    describe_macd(row)
    describe_stochastic(row)
    describe_stock_trend(row)
    describe_moving_averages(row)
    describe_volume(row)
    describe_volatility(row)


def build_forecast_data(
    latest_row,
    calibration,
    current_price,
):
    forecasts = []

    for horizon in HORIZONS:
        probability = predict_horizon(
            latest_row,
            horizon,
        )

        cal = find_calibration(
            calibration,
            horizon,
            probability,
        )

        if cal is None:
            continue

        median_return = float(
            cal["MedianReturn"]
        )

        mean_return = float(
            cal["MeanReturn"]
        )

        p25 = float(
            cal["P25Return"]
        )

        p75 = float(
            cal["P75Return"]
        )

        forecasts.append(
            {
                "horizon": horizon,
                "raw_score": probability,
                "samples": int(
                    cal["Samples"]
                ),
                "actual_up": float(
                    cal["ActualUpRate"]
                ),
                "median_return":
                    median_return,
                "mean_return":
                    mean_return,
                "median_price":
                    current_price
                    * (1 + median_return),
                "mean_price":
                    current_price
                    * (1 + mean_return),
                "low_return":
                    p25,
                "high_return":
                    p75,
                "low_price":
                    current_price
                    * (1 + p25),
                "high_price":
                    current_price
                    * (1 + p75),
            }
        )

    return forecasts


def print_general_summary(
    row,
    forecasts,
):
    print()
    print("=" * 78)
    print("GENEL GÖRÜNÜM")
    print("=" * 78)
    print()

    if not forecasts:
        print(
            "Tahmin verisi bulunamadı."
        )
        return

    forecast_map = {
        item["horizon"]: item
        for item in forecasts
    }

    rsi = float(
        row["rsi_14"]
    )

    market20 = float(
        row["market_return_20d"]
    )

    market60 = float(
        row["market_return_60d"]
    )

    if (
        market20 > 0
        and market60 > 0
    ):
        print(
            "• Genel piyasa hem son haftalarda hem de daha uzun "
            "dönemde yukarı yönlü hareket etmiş."
        )

    elif (
        market20 < 0
        and market60 < 0
    ):
        print(
            "• Genel piyasa hem kısa hem orta vadede "
            "gerileme eğiliminde."
        )

    else:
        print(
            "• Genel piyasanın kısa ve orta vadeli hareketleri "
            "birbirinden farklı; piyasa görünümü karışık."
        )

    if 10 in forecast_map and 30 in forecast_map:
        short_up = np.mean(
            [
                forecast_map[10]["actual_up"],
                forecast_map[30]["actual_up"],
            ]
        )

        short_median = np.mean(
            [
                forecast_map[10]["median_return"],
                forecast_map[30]["median_return"],
            ]
        )

        print(
    f"• 10 günlük benzer geçmiş tahminlerin "
    f"%{forecast_map[10]['actual_up'] * 100:.1f}'si "
    f"yükselişle sonuçlanmış; "
    f"tipik değişim "
    f"{pct(forecast_map[10]['median_return'])}."
)

        print(
    f"• 30 günlük benzer geçmiş tahminlerin "
    f"%{forecast_map[30]['actual_up'] * 100:.1f}'si "
    f"yükselişle sonuçlanmış; "
    f"tipik değişim "
    f"{pct(forecast_map[30]['median_return'])}."
)

    if 60 in forecast_map:
        f60 = forecast_map[60]

        print(
            f"• 60 günlük benzer geçmiş örneklerin "
            f"%{f60['actual_up'] * 100:.1f}'si yükselişle "
            f"sonuçlanmış; tipik değişim "
            f"{pct(f60['median_return'])}."
        )

    if 120 in forecast_map:
        f120 = forecast_map[120]

        print(
            f"• 120 günlük benzer geçmiş örneklerin "
            f"%{f120['actual_up'] * 100:.1f}'si yükselişle "
            f"sonuçlanmış; tipik değişim "
            f"{pct(f120['median_return'])}."
        )

        spread = (
            f120["high_return"]
            - f120["low_return"]
        )

        if spread > 0.40:
            print(
                "  Bu vadedeki geçmiş sonuç aralığı oldukça geniş; "
                "belirsizlik yüksek."
            )

    if 180 in forecast_map:
        f180 = forecast_map[180]

        print(
            f"• 180 günlük benzer geçmiş örneklerde tipik değişim "
            f"{pct(f180['median_return'])}; "
            f"ancak uzun vadede sonuç aralığı daha geniş."
        )

    if rsi >= 70:
        print(
            "• RSI yüksek bölgede; hisse kısa vadede "
            "yorulmuş olabilir."
        )

    elif rsi <= 30:
        print(
            "• RSI düşük bölgede; hisse son dönemde "
            "yoğun satış baskısı yaşamış."
        )

    else:
        print(
            "• RSI aşırı bir bölgede değil."
        )

    if "close_sma_50_ratio" in row.index:
        d50 = float(
            row["close_sma_50_ratio"]
        )

        print(
            f"• Fiyat 50 günlük ortalamanın "
            f"%{abs(d50) * 100:.1f} "
            + (
                "üzerinde."
                if d50 > 0
                else "altında."
            )
        )

    if "close_sma_200_ratio" in row.index:
        d200 = float(
            row["close_sma_200_ratio"]
        )

        print(
            f"• Fiyat 200 günlük ortalamanın "
            f"%{abs(d200) * 100:.1f} "
            + (
                "üzerinde."
                if d200 > 0
                else "altında."
            )
        )


def print_forecast(
    forecast,
):
    horizon = forecast[
        "horizon"
    ]

    print()
    print("=" * 78)
    print(
        f"{horizon} İŞLEM GÜNÜ TAHMİNİ"
    )
    print("=" * 78)

    print()
    print(
        f"Geçmiş benzer tahmin sayısı           : "
        f"{forecast['samples']:,}"
    )

    print(
        f"Geçmişte yükselişle sonuçlanan        : "
        f"%{forecast['actual_up'] * 100:.2f}"
    )

    print()
    print(
        f"Tipik (medyan) değişim                : "
        f"{pct(forecast['median_return'])}"
    )

    print(
        f"Tipik değişime göre tahmini fiyat     : "
        f"~{tl(forecast['median_price'])}"
    )

    print()
    print(
        f"Benzer durumlarda ortalama değişim    : "
        f"{pct(forecast['mean_return'])}"
    )

    print(
        f"Ortalama değişime göre tahmini fiyat  : "
        f"~{tl(forecast['mean_price'])}"
    )

    print()
    print(
        "Geçmiş benzer sonuçların orta %50'si:"
    )

    print(
        f"{pct(forecast['low_return'])} ile "
        f"{pct(forecast['high_return'])} arasında"
    )

    print(
        f"Fiyat karşılığı yaklaşık "
        f"{tl(forecast['low_price'])} – "
        f"{tl(forecast['high_price'])}"
    )

    print()
    print(
        "→ Tahmini fiyatlar geçmişte benzer model sonuçlarında "
        "gerçekleşen fiyat hareketlerine dayanır; "
        "kesin fiyat hedefi değildir."
    )


def main():
    if len(sys.argv) < 2:
        print(
            "Kullanım:"
        )
        print(
            "python src/predict.py THYAO"
        )
        return

    ticker = (
        sys.argv[1]
        .upper()
        .replace(".IS", "")
    )

    try:
        (
            latest_row,
            latest_date,
            latest_close,
        ) = build_latest_row(
            ticker
        )

        calibration = (
            load_calibration()
        )

        forecasts = (
            build_forecast_data(
                latest_row,
                calibration,
                latest_close,
            )
        )

    except Exception as exc:
        print()
        print(
            "HATA:"
        )
        print(
            exc
        )
        return

    row = latest_row.iloc[0]

    print()
    print("=" * 78)
    print(
        f"{ticker} — HİSSE ANALİZİ"
    )
    print("=" * 78)

    print()
    print(
        "Son veri tarihi :",
        latest_date.strftime(
            "%d.%m.%Y"
        ),
    )

    print(
        "Son kapanış     :",
        tl(
            latest_close
        ),
    )

    print_general_summary(
        row,
        forecasts,
    )

    describe_market(
        row
    )

    describe_stock(
        row,
        ticker,
    )

    for forecast in forecasts:
        print_forecast(
            forecast
        )

    print()
    print("=" * 78)
    print("NOT")
    print("=" * 78)

    print()
    print(
        "Bu rapor geçmiş fiyat verileri, teknik göstergeler "
        "ve makine öğrenmesi modellerinden üretilir."
    )

    print(
        "AL / SAT / BEKLE kararı üretmez. "
        "Son değerlendirme kullanıcıya aittir."
    )

    print()


if __name__ == "__main__":
    main()