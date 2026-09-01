"""
BorsaBot Finans — veri ve analiz katmanı.

Streamlit'ten bağımsızdır: fiyat çekme, teknik indikatörler, Gemini asistanı,
ML tahmin köprüsü ve günlük kayıt mantığının tamamı burada. Web katmanı
(web/main.py) yalnızca bu modülü çağırır.
"""

from __future__ import annotations

import functools
import json
import os
import re
import ssl
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from html import escape as html_escape
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from core import daily_log as dl

# yfinance bazı ortamlarda sertifika zinciri kuramıyor
if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ttl_cache(ttl_seconds: int):
    """
    st.cache_data'nın yerine geçen, süreç içi ve thread-safe önbellek.
    st.cache_data gibi DataFrame sonuçlarının kopyasını döndürür ki çağıran
    taraf önbellekteki nesneyi yanlışlıkla değiştiremesin.
    """
    def decorator(fn):
        store: dict = {}
        lock = threading.Lock()

        def _out(value):
            return value.copy() if isinstance(value, pd.DataFrame) else value

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            with lock:
                hit = store.get(key)
                if hit is not None and now - hit[0] < ttl_seconds:
                    return _out(hit[1])
            value = fn(*args, **kwargs)
            with lock:
                store[key] = (now, value)
                if len(store) > 256:
                    for k in [k for k, v in store.items() if now - v[0] >= ttl_seconds]:
                        store.pop(k, None)
            return _out(value)

        wrapper.cache_clear = store.clear
        return wrapper

    return decorator


def _secret_from_file(name: str) -> str:
    """
    .streamlit/secrets.toml'u okumaya devam ediyoruz: sunucuda Gemini anahtarı
    orada duruyor ve dosya .gitignore'da. Deploy akışını değiştirmemek için
    aynı yolu koruyoruz.
    """
    path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not path.exists():
        return ""
    try:
        import tomllib
        with open(path, "rb") as fh:
            return str(tomllib.load(fh).get(name, "") or "")
    except Exception:
        return ""


BIST100_SYMBOLS = [
    "XU030.IS", "XU100.IS",  # Endeksler
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS",
    "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KOZAL.IS", "KOZAA.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
    "SASA.IS", "SISE.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", "TSKB.IS", "TUPRS.IS",
    "VAKBN.IS", "YKBNK.IS", "YUNSA.IS", "ZOREN.IS", "AKSEN.IS", "ALBRK.IS", "ALCAR.IS",
    "ALGYO.IS", "ALKIM.IS", "ASUZU.IS", "AYCES.IS", "BAGFS.IS", "BERA.IS",
    "BIENY.IS", "BRISA.IS", "BRYAT.IS", "BUCIM.IS", "CCOLA.IS", "CEMAS.IS", "CEMTS.IS", "CIMSA.IS",
    "CUSAN.IS", "DOHOL.IS", "EGEEN.IS", "ENJSA.IS", "ENKAI.IS", "FMIZP.IS", "GESAN.IS", "GLYHO.IS",
    "GSDHO.IS", "GSDDE.IS", "HALKB.IS", "HATEK.IS", "IPEKE.IS", "ISDMR.IS", "KAREL.IS", "KARSN.IS",
    "KONTR.IS", "KONYA.IS", "KORDS.IS", "KRDMD.IS", "LOGO.IS", "MGROS.IS",
    "NTHOL.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "POLHO.IS", "PRKAB.IS",
    "PRKME.IS", "QUAGR.IS", "SELEC.IS", "SELGD.IS", "SKBNK.IS", "SMRTG.IS",
    "SNGYO.IS", "SOKM.IS", "TATGD.IS", "TTKOM.IS",
    "TTRAK.IS", "ULKER.IS", "VESTL.IS", "YATAS.IS",
]


MARKET_TABS = {
    "BIST": [
        ("XU100.IS", "BIST 100"),
        ("XU030.IS", "BIST 30"),
        ("XBANK.IS", "Bankacılık"),
        ("XUSIN.IS", "Sınai"),
        ("XUTEK.IS", "Teknoloji"),
        ("XUMAL.IS", "Mali"),
    ],
    "Döviz": [
        ("USDTRY=X", "Dolar/TL"),
        ("EURTRY=X", "Euro/TL"),
        ("GBPTRY=X", "Sterlin/TL"),
        ("EURUSD=X", "Euro/Dolar"),
        ("CHFTRY=X", "Frank/TL"),
        ("DX-Y.NYB", "Dolar Endeksi"),
    ],
    "Emtia": [
        ("GC=F", "Ons Altın"),
        ("SI=F", "Gümüş"),
        ("BZ=F", "Brent Petrol"),
        ("CL=F", "Ham Petrol"),
        ("NG=F", "Doğal Gaz"),
        ("HG=F", "Bakır"),
    ],
    "Kripto": [
        ("BTC-USD", "Bitcoin"),
        ("ETH-USD", "Ethereum"),
        ("SOL-USD", "Solana"),
        ("XRP-USD", "XRP"),
        ("AVAX-USD", "Avalanche"),
        ("DOGE-USD", "Dogecoin"),
    ],
    "Küresel": [
        ("^GSPC", "S&P 500"),
        ("^IXIC", "Nasdaq"),
        ("^DJI", "Dow Jones"),
        ("^GDAXI", "DAX"),
        ("^FTSE", "FTSE 100"),
        ("^N225", "Nikkei 225"),
    ],
    "Vadeli": [
        ("ES=F", "S&P 500 Vadeli"),
        ("NQ=F", "Nasdaq Vadeli"),
        ("YM=F", "Dow Vadeli"),
        ("GC=F", "Altın Vadeli"),
        ("CL=F", "Petrol Vadeli"),
        ("ZW=F", "Buğday Vadeli"),
    ],
}


SECTOR_INDICES = [
    ("XBANK.IS", "Bankacılık"),
    ("XUSIN.IS", "Sınai"),
    ("XUTEK.IS", "Teknoloji"),
    ("XUHIZ.IS", "Hizmetler"),
    ("XGIDA.IS", "Gıda İçecek"),
    ("XHOLD.IS", "Holding"),
    ("XKMYA.IS", "Kimya Petrol"),
    ("XILTM.IS", "İletişim"),
]


TICKER_STRIP = [
    ("XU100.IS", "BIST 100"),
    ("USDTRY=X", "Dolar"),
    ("EURTRY=X", "Euro"),
    ("GC=F", "Ons Altın"),
    ("BZ=F", "Brent"),
    ("BTC-USD", "Bitcoin"),
]


def _series_for(df, ticker):
    """yf.download çıktısından tek bir ticker'ın Close serisini güvenle çıkarır."""
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(0):
                sub = df[ticker]
            elif ticker in df.columns.get_level_values(1):
                sub = df.xs(ticker, axis=1, level=1)
            else:
                return None
            close = sub['Close']
        else:
            close = df['Close']
        close = pd.to_numeric(close, errors='coerce').dropna()
        return close if len(close) else None
    except Exception:
        return None


_SPARK_URL = "https://query1.finance.yahoo.com/v7/finance/spark"


_SPARK_CHUNK = 20


_SPARK_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


_PERIOD_TO_RANGE = {
    "1d": "1d", "5d": "5d", "1mo": "1mo", "3mo": "3mo", "6mo": "6mo",
    "1y": "1y", "2y": "2y", "5y": "5y", "10y": "10y", "ytd": "ytd", "max": "max",
}


def _spark_fetch(chunk: tuple, rng: str, interval: str) -> list:
    resp = requests.get(
        _SPARK_URL,
        params={"symbols": ",".join(chunk), "range": rng, "interval": interval},
        headers=_SPARK_HEADERS,
        timeout=12,
    )
    resp.raise_for_status()
    return (resp.json().get("spark") or {}).get("result") or []


def _spark_quotes(tickers: list, period: str, interval: str) -> dict:
    """Spark uç noktasından {ticker: {price, change, pct, spark}} döndürür."""
    rng = _PERIOD_TO_RANGE.get(period, "1mo")
    chunks = [tuple(tickers[i:i + _SPARK_CHUNK]) for i in range(0, len(tickers), _SPARK_CHUNK)]
    results = []
    with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as ex:
        futures = [ex.submit(_spark_fetch, c, rng, interval) for c in chunks]
        for f in futures:
            try:
                results.extend(f.result())
            except Exception:
                continue

    out = {}
    for item in results:
        try:
            symbol = item.get("symbol")
            resp = (item.get("response") or [None])[0]
            if not symbol or not resp:
                continue
            quote = ((resp.get("indicators") or {}).get("quote") or [{}])[0]
            closes = [float(v) for v in (quote.get("close") or []) if v is not None]
            meta = resp.get("meta") or {}
            if not closes:
                price = meta.get("regularMarketPrice")
                if price is None:
                    continue
                closes = [float(price)]
            price = closes[-1]
            prev = closes[-2] if len(closes) > 1 else meta.get("chartPreviousClose") or price
            prev = float(prev)
            change = price - prev
            out[symbol] = {
                "price": price,
                "change": change,
                "pct": (change / prev * 100) if prev else 0.0,
                "spark": closes[-40:],
            }
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _yf_quotes(tickers: list, period: str, interval: str) -> dict:
    """Yedek yol: spark uç noktası yanıt vermezse yfinance ile indir."""
    out = {}
    try:
        raw = yf.download(
            tickers, period=period, interval=interval,
            group_by='ticker', progress=False, auto_adjust=True, threads=True,
        )
    except Exception:
        return {}

    for t in tickers:
        close = _series_for(raw, t)
        if close is None or len(close) == 0:
            continue
        price = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else price
        change = price - prev
        out[t] = {
            "price": price,
            "change": change,
            "pct": (change / prev * 100) if prev else 0.0,
            "spark": [float(v) for v in close.tail(40).tolist()],
        }
    return out


@ttl_cache(180)
def get_quotes(tickers: tuple, period: str = "1mo", interval: str = "1d") -> dict:
    """
    Verilen semboller için {ticker: {price, change, pct, spark}} döndürür.
    spark: mini grafik için son kapanış listesi.
    """
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return {}

    out = _spark_quotes(tickers, period, interval)
    if not out:
        # Spark hiç veri döndürmediyse (uç nokta erişilemiyor) eski yola düş.
        out = _yf_quotes(tickers, period, interval)
    return out


def home_ticker_universe(extra: tuple = ()) -> tuple:
    """Ana sayfanın tek bir istekte çekeceği tüm sembollerin birleşimi."""
    syms = [t for t, _ in TICKER_STRIP]
    syms += [t for t, _ in SECTOR_INDICES]
    for items in MARKET_TABS.values():
        syms += [t for t, _ in items]
    syms += [s for s in BIST100_SYMBOLS if not s.startswith("XU")][:MOVERS_LIMIT]
    syms += list(extra)
    return tuple(dict.fromkeys(syms))


def prefetch_home_quotes(extra: tuple = ()) -> dict:
    """
    Ana sayfadaki tüm bölümlerin verisini TEK istek turunda çeker.
    Bölümler ayrı ayrı get_quotes çağırdığında her biri ayrı bir HTTP turu
    demekti (10 × ~0.4 sn); burada hepsi tek seferde alınır.
    """
    return get_quotes(home_ticker_universe(extra))


MOVERS_LIMIT = 40


def get_market_movers(quotes: dict | None = None, limit: int = MOVERS_LIMIT) -> pd.DataFrame:
    """En çok yükselen / düşen hisseler."""
    tickers = tuple(s for s in BIST100_SYMBOLS if not s.startswith("XU"))[:limit]
    if quotes is None:
        quotes = get_quotes(tickers)
    rows = []
    for t in tickers:
        q = quotes.get(t)
        if not q:
            continue
        rows.append({"Sembol": t.replace(".IS", ""), "Fiyat": q["price"], "Değişim %": q["pct"]})
    if not rows:
        return pd.DataFrame(columns=["Sembol", "Fiyat", "Değişim %"])
    return pd.DataFrame(rows).sort_values("Değişim %", ascending=False).reset_index(drop=True)


@ttl_cache(300)
def get_bloomberg_news():
    RSS_URL = "https://www.bloomberght.com/rss"
    try:
        response = requests.get(RSS_URL, timeout=10)
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')
        news_list = []
        for item in items:
            title = item.find('title').text if item.find('title') else "Başlık Yok"
            link = item.find('link').text if item.find('link') else "#"
            pub_date = item.find('pubDate').text if item.find('pubDate') else ""
            news_list.append({'title': title, 'link': link, 'published': pub_date})
        return news_list
    except Exception:
        return []


@ttl_cache(300)
def get_stock_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Yahoo, seans kapanmadan tamamı boş bir satır döndürebiliyor. Bu satır
        # mum grafiğinde ekseni sıfıra çekiyor ve indikatörleri NaN yapıyor.
        if "Close" in df.columns:
            df = df[df["Close"].notna()]
        return df
    except Exception:
        return None


def fmt_price(v: float) -> str:
    """Google Finance tarzı sayı biçimi (binlik ayraç nokta, ondalık virgül)."""
    if v is None or pd.isna(v):
        return "-"
    dec = 2 if abs(v) >= 1 else 4
    s = f"{v:,.{dec}f}"
    return s.replace(",", "@").replace(".", ",").replace("@", ".")


def trend_class(pct: float) -> str:
    if pct is None or pd.isna(pct):
        return "gf-flat"
    if pct > 0:
        return "gf-up"
    if pct < 0:
        return "gf-down"
    return "gf-flat"


def trend_arrow(pct: float) -> str:
    if pct is None or pd.isna(pct):
        return ""
    return "▲" if pct > 0 else ("▼" if pct < 0 else "▬")


_spark_id = 0


def sparkline_svg(values, pct, width=160, height=44):
    """Google Finance kartlarındaki mini alan grafiğini SVG olarak üretir."""
    if not values or len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    n = len(values)
    color = "#81c995" if pct >= 0 else "#f28b82"
    pts = []
    for i, v in enumerate(values):
        x = i * (width - 2) / (n - 1) + 1
        y = height - 4 - ((v - lo) / rng) * (height - 8)
        pts.append(f"{x:.1f},{y:.1f}")
    line = " ".join(pts)
    area = f"1,{height} {line} {width - 1},{height}"
    global _spark_id
    _spark_id += 1
    uid = f"gfspark{_spark_id}"
    return f"""<svg class="gf-spark" width="100%" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="none">
<defs><linearGradient id="{uid}" x1="0" x2="0" y1="0" y2="1">
<stop offset="0%" stop-color="{color}" stop-opacity="0.28"/>
<stop offset="100%" stop-color="{color}" stop-opacity="0"/>
</linearGradient></defs>
<polygon points="{area}" fill="url(#{uid})"/>
<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.5"
 stroke-linejoin="round" stroke-linecap="round"/></svg>"""


def market_card_html(name: str, q: dict) -> str:
    pct = q["pct"]
    cls = trend_class(pct)
    sign = "+" if q["change"] >= 0 else ""
    return f"""<div class="gf-card">
  <div class="gf-card-name">{name}</div>
  <div class="gf-card-price">{fmt_price(q['price'])}</div>
  <div class="gf-card-abs">({sign}{fmt_price(q['change'])})</div>
  <div class="gf-card-chg {cls}">{sign}{pct:.2f}% {trend_arrow(pct)}</div>
  {sparkline_svg(q['spark'], pct)}
</div>"""


def quote_row_html(sym: str, name: str, q: dict) -> str:
    pct = q["pct"]
    cls = trend_class(pct)
    sign = "+" if pct >= 0 else ""
    return f"""<div class="gf-row">
  <div class="gf-row-left">
    <span class="gf-row-sym">{sym}</span>
    <span class="gf-row-name">{name}</span>
  </div>
  <div class="gf-row-right">
    <span class="gf-row-price">{fmt_price(q['price'])}</span>
    <span class="gf-row-chg {cls}">{sign}{pct:.2f}%</span>
  </div>
</div>"""


def signal_badge(sig: str) -> str:
    s = (sig or "").upper()
    if "AL" in s and "SAT" not in s:
        return f'<span class="gf-badge gf-badge-buy">{sig}</span>'
    if "SAT" in s:
        return f'<span class="gf-badge gf-badge-sell">{sig}</span>'
    return f'<span class="gf-badge gf-badge-wait">{sig or "BEKLE"}</span>'


def safe_link(url: str) -> str:
    """
    Yalnızca http/https bağlantılarına izin verir.
    Streamlit'in unsafe_allow_html yolu URL şemasını temizlemiyor; html_escape
    attribute'tan çıkışı engelliyor ama "javascript:" gibi bir şemayı olduğu
    gibi geçiriyor. RSS gibi dış kaynaklardan gelen bağlantılar için gerekli.
    """
    if isinstance(url, str) and url.strip().lower().startswith(("http://", "https://")):
        return url.strip()
    return "#"


def price_levels(df: pd.DataFrame) -> dict:
    """
    Asistanın somut konuşabilmesi için teknik fiyat seviyeleri:
    hareketli ortalamalar, Bollinger, pivot, ATR, swing ve Fibonacci seviyeleri.
    """
    out = {}
    try:
        close = pd.to_numeric(df['Close'], errors='coerce')
        high = pd.to_numeric(df['High'], errors='coerce')
        low = pd.to_numeric(df['Low'], errors='coerce')
        last = float(close.iloc[-1])
        out["son"] = last

        for w in (20, 50, 200):
            v = close.rolling(w).mean().iloc[-1]
            if not pd.isna(v):
                out[f"ma{w}"] = float(v)

        sma20 = close.rolling(20).mean().iloc[-1]
        std20 = close.rolling(20).std().iloc[-1]
        if not pd.isna(sma20) and not pd.isna(std20):
            out["bb_ust"] = float(sma20 + 2 * std20)
            out["bb_orta"] = float(sma20)
            out["bb_alt"] = float(sma20 - 2 * std20)

        # ATR(14): günlük tipik hareket
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        if not pd.isna(atr):
            out["atr"] = float(atr)
            out["atr_yuzde"] = float(atr / last * 100)

        # Klasik pivot (son bar)
        h, l, c = float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
        p = (h + l + c) / 3
        out["pivot"] = p
        out["r1"], out["s1"] = 2 * p - l, 2 * p - h
        out["r2"], out["s2"] = p + (h - l), p - (h - l)

        # Swing seviyeleri
        for w, key in ((20, "20g"), (60, "60g")):
            out[f"tepe_{key}"] = float(high.rolling(w).max().iloc[-1])
            out[f"dip_{key}"] = float(low.rolling(w).min().iloc[-1])

        # Son 6 ayın Fibonacci düzeltmeleri
        seg = df.tail(126)
        sh = float(pd.to_numeric(seg['High'], errors='coerce').max())
        sl = float(pd.to_numeric(seg['Low'], errors='coerce').min())
        diff = sh - sl
        if diff > 0:
            out["fib_tepe"], out["fib_dip"] = sh, sl
            for r in (0.236, 0.382, 0.5, 0.618, 0.786):
                out[f"fib_{r}"] = sh - r * diff

        # Fiyata en yakın destek / dirençler
        adaylar = []
        etiket = {
            "ma20": "MA20", "ma50": "MA50", "ma200": "MA200",
            "bb_ust": "Bollinger üst", "bb_alt": "Bollinger alt", "bb_orta": "Bollinger orta",
            "r1": "Pivot R1", "r2": "Pivot R2", "s1": "Pivot S1", "s2": "Pivot S2",
            "tepe_20g": "20 günlük tepe", "dip_20g": "20 günlük dip",
            "tepe_60g": "60 günlük tepe", "dip_60g": "60 günlük dip",
            "fib_0.236": "Fib %23.6", "fib_0.382": "Fib %38.2", "fib_0.5": "Fib %50",
            "fib_0.618": "Fib %61.8", "fib_0.786": "Fib %78.6",
        }
        for key, name in etiket.items():
            if key in out:
                adaylar.append((name, out[key]))
        out["destekler"] = sorted([a for a in adaylar if a[1] < last], key=lambda x: -x[1])[:4]
        out["dirençler"] = sorted([a for a in adaylar if a[1] > last], key=lambda x: x[1])[:4]
    except Exception:
        pass
    return out


@ttl_cache(300)
def get_analysis_context(symbol: str):
    """
    Bir hisse için popup ve AI asistanının kullandığı ortak teknik veri paketi.
    Döndürür: {'quote':..., 'signals': DataFrame, 'genel': str, 'text': str} veya None
    """
    df = get_stock_data(symbol, "1y", "1d")
    if df is None or df.empty:
        return None

    signals_df = calculate_technical_signals(df)
    sig_map = _signals_df_to_dict(signals_df)
    genel = _summarize_general(sig_map) if sig_map else "BEKLE"

    close = pd.to_numeric(df['Close'], errors='coerce').dropna()
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last
    pct = ((last - prev) / prev * 100) if prev else 0.0
    quote = {
        "price": last,
        "change": last - prev,
        "pct": pct,
        "spark": [float(v) for v in close.tail(40).tolist()],
    }

    try:
        hi52 = float(pd.to_numeric(df['High'], errors='coerce').max())
        lo52 = float(pd.to_numeric(df['Low'], errors='coerce').min())
        vol = pd.to_numeric(df['Volume'], errors='coerce').dropna()
        vol_last = float(vol.iloc[-1]) if len(vol) else 0.0
        vol_avg = float(vol.tail(20).mean()) if len(vol) else 0.0
    except Exception:
        hi52 = lo52 = vol_last = vol_avg = 0.0

    lines = [
        f"Sembol: {symbol}",
        f"Son kapanış: {last:.2f} TL (günlük değişim: {pct:+.2f}%)",
        f"Son 1 yıl aralığı: {lo52:.2f} - {hi52:.2f}",
        f"Son hacim: {vol_last:,.0f} / 20 günlük ortalama hacim: {vol_avg:,.0f}",
        f"Uygulamanın genel teknik görünümü: {genel}",
        "İndikatörler:",
    ]
    for _, r in signals_df.iterrows():
        lines.append(f"- {r['İndikatör']}: sinyal={r['Sinyal']} | {r['Değerler']}")

    lv = price_levels(df)
    if lv:
        lines.append("")
        lines.append("Teknik fiyat seviyeleri (TL):")
        for key, label in (
            ("ma20", "MA20"), ("ma50", "MA50"), ("ma200", "MA200"),
            ("bb_ust", "Bollinger üst"), ("bb_orta", "Bollinger orta"), ("bb_alt", "Bollinger alt"),
            ("pivot", "Pivot"), ("r1", "Pivot R1"), ("r2", "Pivot R2"),
            ("s1", "Pivot S1"), ("s2", "Pivot S2"),
            ("tepe_20g", "20 günlük tepe"), ("dip_20g", "20 günlük dip"),
            ("tepe_60g", "60 günlük tepe"), ("dip_60g", "60 günlük dip"),
            ("fib_0.236", "Fib %23.6"), ("fib_0.382", "Fib %38.2"), ("fib_0.5", "Fib %50"),
            ("fib_0.618", "Fib %61.8"), ("fib_0.786", "Fib %78.6"),
        ):
            if key in lv:
                lines.append(f"- {label}: {lv[key]:.2f}")
        if "atr" in lv:
            lines.append(
                f"- ATR(14) günlük tipik hareket: {lv['atr']:.2f} TL (%{lv['atr_yuzde']:.2f})"
            )
        if lv.get("destekler"):
            lines.append("- Fiyatın altındaki en yakın destekler: " + ", ".join(
                f"{n} {v:.2f}" for n, v in lv["destekler"]))
        if lv.get("dirençler"):
            lines.append("- Fiyatın üstündeki en yakın dirençler: " + ", ".join(
                f"{n} {v:.2f}" for n, v in lv["dirençler"]))

    return {
        "quote": quote, "signals": signals_df, "genel": genel,
        "levels": lv, "text": "\n".join(lines),
    }


GEMINI_MODEL = "gemini-2.5-flash"


GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


AI_SYSTEM_PROMPT = """Sen BorsaBot Finans uygulamasının teknik analiz asistanısın.
Karşındaki kullanıcı deneyimli bir yatırımcı ve senden LAF KALABALIĞI DEĞİL, SOMUT FİYAT SEVİYELERİ bekliyor.

Kurallar:
- SADECE sana verilen fiyat, seviye ve indikatör verileri üzerinden konuş. Veri yoksa "elimde bu veri yok" de, uydurma.
- Aşağıda birden fazla hissenin verisi olabilir; kullanıcı hangisini sorduysa onun bölümünü kullan,
  iki hisse sorulduysa ikisini karşılaştır.
- SOMUT OL. "Destek var" deme; "384,00 (Fib %23.6) ilk destek" de. Her seviyeyi rakamla ver ve
  hangi göstergeden geldiğini parantezde yaz. Yuvarlak laf, genel geçer cümle kurma.
- Şu başlıkları kullan (gereksizini atla):
  • Görünüm — trend ve momentum tek cümlede, sayılarla.
  • İzlenecek destekler — 2-3 seviye, rakam + kaynak.
  • İzlenecek dirençler — 2-3 seviye, rakam + kaynak.
  • Senaryolar — "X üstünde kalıcı olursa ilk hedef Y", "Z altına sarkarsa yapı bozulur, sıradaki bölge W".
  • Risk — yapının bozulduğu seviye ve ATR'ye göre günlük tipik hareket.
- Teknik olarak öne çıkan alım/kâr realizasyonu BÖLGELERİNİ seviye olarak söyleyebilirsin:
  "382-388 bandı teknik olarak tepki alımı bölgesi olarak izlenir", "426 Bollinger üst bandı
  kâr realizasyonunun yoğunlaştığı bölge" gibi. Bunlar teknik gözlemdir.
- Ancak kullanıcıya kişisel emir verme: "hemen al", "sat", "şu kadar lot gir", "portföyünün %X'i"
  gibi ifadeler kullanma. Kesinlik iddia etme ("kesin çıkar", "garanti") — olasılık dilinde konuş.
- Pozisyon büyüklüğü, kaldıraç, portföy dağılımı ve kişisel finansal durum konularına hiç girme.
- Uygulamadaki AL/SAT sinyalleri indikatörlerin mekanik çıktısıdır; "uygulamanın MACD sinyali AL
  üretiyor" gibi teknik bir gözlem olarak aktar.
- Türkçe, kısa ve yoğun yaz: en fazla 200 kelime, madde madde, giriş cümlesi kurmadan.
- Cevabın sonuna tek satır ekle: "Not: Bu bir teknik analiz yorumudur, yatırım tavsiyesi değildir."
"""


BIST_ALIASES = {
    "aselsan": "ASELS", "türk hava yolları": "THYAO", "turk hava yollari": "THYAO",
    "thy": "THYAO", "akbank": "AKBNK", "garanti": "GARAN", "iş bankası": "ISCTR",
    "is bankasi": "ISCTR", "işbank": "ISCTR", "yapı kredi": "YKBNK", "yapi kredi": "YKBNK",
    "vakıfbank": "VAKBN", "vakifbank": "VAKBN", "halkbank": "HALKB", "şişecam": "SISE",
    "sisecam": "SISE", "ereğli": "EREGL", "eregli": "EREGL", "erdemir": "EREGL",
    "tüpraş": "TUPRS", "tupras": "TUPRS", "ford otosan": "FROTO", "tofaş": "TOASO",
    "tofas": "TOASO", "koç holding": "KCHOL", "koc holding": "KCHOL", "sabancı": "SAHOL",
    "sabanci": "SAHOL", "bim": "BIMAS", "migros": "MGROS", "turkcell": "TCELL",
    "türk telekom": "TTKOM", "turk telekom": "TTKOM", "pegasus": "PGSUS",
    "petkim": "PETKM", "arçelik": "ARCLK", "arcelik": "ARCLK", "ülker": "ULKER",
    "ulker": "ULKER", "vestel": "VESTL", "enka": "ENKAI", "hektaş": "HEKTS",
    "hektas": "HEKTS", "emlak konut": "EKGYO", "gübretaş": "GUBRF", "gubretas": "GUBRF",
    "oyak çimento": "OYAKC", "koza altın": "KOZAL", "koza altin": "KOZAL",
    "bist 100": "XU100", "bist100": "XU100", "bist 30": "XU030", "bist30": "XU030",
}


def detect_symbols(text: str, exclude: str | None = None) -> list:
    """Kullanıcının sorusunda geçen hisseleri bulur (kod ya da şirket adı)."""
    t = (text or "").lower()
    found = []

    for name, code in BIST_ALIASES.items():
        # "sahibim" içindeki "bim" gibi yanlış eşleşmeleri önlemek için kelime sınırı
        if re.search(rf"\b{re.escape(name)}\b", t):
            found.append(code)

    known = {s.replace(".IS", "") for s in BIST100_SYMBOLS}
    known.update(BIST_ALIASES.values())
    for code in known:
        if re.search(rf"\b{code.lower()}\b", t):
            found.append(code)

    out = []
    for code in found:
        sym = f"{code}.IS"
        if sym == exclude or sym in out:
            continue
        out.append(sym)
    return out


GEMINI_RATE_LIMIT = 10


GEMINI_RATE_WINDOW = 60


GEMINI_DAILY_LIMIT = 200


_gemini_lock = threading.Lock()


_gemini_usage = {"times": deque(), "day": None, "count": 0}


def _gemini_prune(now: float, today: str):
    """Kayan pencereden eskiyen istekleri ve dün kalan günlük sayacı temizler."""
    times = _gemini_usage["times"]
    while times and now - times[0] >= GEMINI_RATE_WINDOW:
        times.popleft()
    if _gemini_usage["day"] != today:
        _gemini_usage["day"] = today
        _gemini_usage["count"] = 0


def gemini_usage() -> dict:
    """Kalan hakları döndürür (arayüzde göstermek için)."""
    now = time.monotonic()
    today = dl.today_istanbul().isoformat()
    with _gemini_lock:
        _gemini_prune(now, today)
        return {
            "minute_left": max(0, GEMINI_RATE_LIMIT - len(_gemini_usage["times"])),
            "daily_left": max(0, GEMINI_DAILY_LIMIT - _gemini_usage["count"]),
        }


def _gemini_acquire() -> tuple[bool, str]:
    """
    Limit içindeyse isteği kaydeder ve (True, "") döner.
    Aşıldıysa (False, kullanıcıya gösterilecek mesaj) döner.
    """
    now = time.monotonic()
    today = dl.today_istanbul().isoformat()
    with _gemini_lock:
        _gemini_prune(now, today)

        if _gemini_usage["count"] >= GEMINI_DAILY_LIMIT:
            return False, (
                f"Günlük Gemini limiti doldu ({GEMINI_DAILY_LIMIT} soru). "
                "Türkiye saatiyle gece yarısı sıfırlanır."
            )

        times = _gemini_usage["times"]
        if len(times) >= GEMINI_RATE_LIMIT:
            wait = int(GEMINI_RATE_WINDOW - (now - times[0])) + 1
            return False, (
                f"Çok hızlı soru soruyorsun (dakikada en fazla {GEMINI_RATE_LIMIT}). "
                f"{wait} saniye sonra tekrar dene."
            )

        times.append(now)
        _gemini_usage["count"] += 1
        return True, ""


def get_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "") or _secret_from_file("GEMINI_API_KEY")
    return (key or "").strip()


def ask_gemini(history: list, context_text: str):
    """Gemini'ye teknik veri bağlamıyla soru sorar. (cevap, hata) döndürür."""
    key = get_gemini_key()
    if not key:
        return None, "Gemini API anahtarı tanımlı değil."

    allowed, limit_msg = _gemini_acquire()
    if not allowed:
        return None, limit_msg

    system_prompt = AI_SYSTEM_PROMPT + "\n\nGüncel teknik veriler:\n" + context_text

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {"role": ("user" if m["role"] == "user" else "model"), "parts": [{"text": m["content"]}]}
            for m in history[-10:]
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 800,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    try:
        r = requests.post(
            GEMINI_URL,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=payload,
            timeout=45,
        )
    except Exception as e:
        return None, f"Bağlantı hatası: {e}"

    if r.status_code != 200:
        detail = ""
        try:
            detail = r.json().get("error", {}).get("message", "")
        except Exception:
            detail = r.text[:180]
        return None, f"Gemini hatası ({r.status_code}): {detail}"

    try:
        parts = r.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
        return (text or None), (None if text else "Boş yanıt döndü.")
    except Exception:
        return None, "Yanıt çözümlenemedi."


# Modül core/ altında olduğu için proje köküne göre çözülür
FORECASTER_DIR = str(PROJECT_ROOT / "stock_forecaster")


FORECASTER_VENV_PY = os.path.join(FORECASTER_DIR, ".venv", "bin", "python")


FORECASTER_SCRIPT = os.path.join("src", "predict_json.py")


HORIZON_LABELS = {
    10: "10 işlem günü (~2 hafta)",
    30: "30 işlem günü (~1,5 ay)",
    60: "60 işlem günü (~3 ay)",
    120: "120 işlem günü (~6 ay)",
    180: "180 işlem günü (~9 ay)",
}


def _forecaster_python() -> str:
    """stock_forecaster kendi venv'iyle çalışır (sklearn/numpy sürümleri farklı)."""
    return FORECASTER_VENV_PY if os.path.exists(FORECASTER_VENV_PY) else sys.executable


def forecaster_available() -> bool:
    return os.path.exists(os.path.join(FORECASTER_DIR, FORECASTER_SCRIPT))


def _run_forecaster(args: list, timeout: int = 300):
    try:
        proc = subprocess.run(
            [_forecaster_python(), FORECASTER_SCRIPT] + args,
            cwd=FORECASTER_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "Model zaman aşımına uğradı."
    except Exception as e:
        return None, f"Model çalıştırılamadı: {e}"

    out = (proc.stdout or "").strip()
    if not out:
        return None, (proc.stderr or "Model çıktı üretmedi.").strip()[-300:]
    try:
        data = json.loads(out.splitlines()[-1])
    except Exception:
        return None, f"Model çıktısı okunamadı: {out[:200]}"
    if "error" in data:
        return None, data["error"]
    return data, None


@ttl_cache(3600)
def ml_supported_tickers() -> list:
    if not forecaster_available():
        return []
    data, _ = _run_forecaster(["--list"], timeout=60)
    return data.get("tickers", []) if data else []


@ttl_cache(1800)
def ml_forecast(ticker: str):
    """Bir hisse için ML tahmini üretir. (sonuç, hata) döndürür."""
    return _run_forecaster([ticker.replace(".IS", "")], timeout=300)


def update_forecaster_data():
    """Modelin kullandığı ham fiyat verisini yeniler (hisseler + BIST 100)."""
    py = _forecaster_python()
    logs = []
    for script in ("src/download_all.py", "src/download_market.py"):
        try:
            proc = subprocess.run(
                [py, script], cwd=FORECASTER_DIR,
                capture_output=True, text=True, timeout=900,
            )
            logs.append(f"{script}: {'tamam' if proc.returncode == 0 else 'hata'}")
            if proc.returncode != 0:
                logs.append((proc.stderr or "")[-300:])
        except Exception as e:
            logs.append(f"{script}: {e}")
    return logs


def _forecast_card_html(f: dict) -> str:
    med = f["median_return"]
    cls = trend_class(med * 100)
    sign = "+" if med >= 0 else ""
    label = HORIZON_LABELS.get(f["horizon"], f"{f['horizon']} gün")
    return f"""<div class="gf-card" style="flex:1 0 210px;min-width:210px;">
  <div class="gf-card-name">{label}</div>
  <div class="gf-card-price">{fmt_price(f['median_price'])}</div>
  <div class="gf-card-abs">tipik (medyan) beklenti</div>
  <div class="gf-card-chg {cls}">{sign}{med * 100:.2f}% {trend_arrow(med)}</div>
  <div style="margin-top:10px;border-top:1px solid var(--gf-border);padding-top:8px;">
    <div class="gf-source">Benzer geçmişte yükseliş oranı</div>
    <div style="font-size:.95rem;margin-top:2px;">%{f['actual_up'] * 100:.1f}
      <span class="gf-source">({f['samples']:,} örnek)</span></div>
    <div class="gf-source" style="margin-top:8px;">Olası aralık (P25 – P75)</div>
    <div style="font-size:.9rem;margin-top:2px;">{fmt_price(f['low_price'])} – {fmt_price(f['high_price'])}</div>
    <div class="gf-source" style="margin-top:8px;">Model yön skoru: %{f['raw_score'] * 100:.1f}</div>
  </div>
</div>"""


def calculate_technical_signals(df):
    if df is None or df.empty:
        return pd.DataFrame()

    last_close = df['Close'].iloc[-1]
    signals = []

    # Helper for signal string
    def get_signal(condition_buy, condition_sell):
        if condition_buy: return "AL"
        if condition_sell: return "SAT"
        return "BEKLE"

    # --- MA ---
    try:
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        ma50 = df['Close'].rolling(50).mean().iloc[-1]
        ma200 = df['Close'].rolling(200).mean().iloc[-1]
        
        ma_signal = "BEKLE"
        if last_close > ma20 and last_close > ma50: ma_signal = "AL"
        elif last_close < ma20 and last_close < ma50: ma_signal = "SAT"
        
        signals.append({
            "İndikatör": "Hareketli Ortalamalar (MA)",
            "Sinyal": ma_signal,
            "Değerler": f"Fiyat: {last_close:.2f}, MA20: {ma20:.2f}, MA50: {ma50:.2f}, MA200: {ma200:.2f}"
        })
    except: pass

    # --- MACD ---
    try:
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        curr_macd = macd.iloc[-1]
        curr_signal = signal.iloc[-1]
        
        macd_sig = get_signal(curr_macd > curr_signal, curr_macd < curr_signal)
        signals.append({
            "İndikatör": "MACD",
            "Sinyal": macd_sig,
            "Değerler": f"MACD: {curr_macd:.2f}, Signal: {curr_signal:.2f}"
        })
    except: pass

    # --- RSI ---
    try:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = rsi.iloc[-1]
        
        rsi_sig = "BEKLE"
        if curr_rsi < 30: rsi_sig = "GÜÇLÜ AL"
        elif curr_rsi > 70: rsi_sig = "GÜÇLÜ SAT"
        
        signals.append({
            "İndikatör": "RSI",
            "Sinyal": rsi_sig,
            "Değerler": f"RSI: {curr_rsi:.2f}"
        })
    except: pass

    # --- Bollinger ---
    try:
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        
        curr_upper = upper.iloc[-1]
        curr_lower = lower.iloc[-1]
        curr_sma = sma20.iloc[-1]
        
        bb_sig = "BEKLE"
        if last_close < curr_lower: bb_sig = "AL (Tepki)"
        elif last_close > curr_upper: bb_sig = "SAT (Tepki)"
        
        signals.append({
            "İndikatör": "Bollinger Bantları",
            "Sinyal": bb_sig,
            "Değerler": f"Üst: {curr_upper:.2f}, Alt: {curr_lower:.2f}, Orta: {curr_sma:.2f}"
        })
    except: pass

    # --- Stochastic ---
    try:
        low14 = df['Low'].rolling(window=14).min()
        high14 = df['High'].rolling(window=14).max()
        k = 100 * ((df['Close'] - low14) / (high14 - low14))
        d = k.rolling(window=3).mean()
        
        curr_k = k.iloc[-1]
        curr_d = d.iloc[-1]
        
        stoch_sig = "BEKLE"
        if curr_k < 20 and curr_d < 20 and curr_k > curr_d: stoch_sig = "AL"
        elif curr_k > 80 and curr_d > 80 and curr_k < curr_d: stoch_sig = "SAT"
        
        signals.append({
            "İndikatör": "Stokastik",
            "Sinyal": stoch_sig,
            "Değerler": f"K: {curr_k:.2f}, D: {curr_d:.2f}"
        })
    except: pass

    # --- Ichimoku ---
    try:
        high9 = df['High'].rolling(window=9).max()
        low9 = df['Low'].rolling(window=9).min()
        tenkan = (high9 + low9) / 2
        
        high26 = df['High'].rolling(window=26).max()
        low26 = df['Low'].rolling(window=26).min()
        kijun = (high26 + low26) / 2
        
        curr_tenkan = tenkan.iloc[-1]
        curr_kijun = kijun.iloc[-1]
        
        ichi_sig = get_signal(curr_tenkan > curr_kijun, curr_tenkan < curr_kijun)
        signals.append({
            "İndikatör": "Ichimoku",
            "Sinyal": ichi_sig,
            "Değerler": f"Tenkan: {curr_tenkan:.2f}, Kijun: {curr_kijun:.2f}"
        })
    except: pass

    # --- Volatility (Std Dev) ---
    try:
        curr_std = df['Close'].rolling(20).std().iloc[-1]
        signals.append({
            "İndikatör": "Standart Sapma (Volatilite)",
            "Sinyal": "N/A",
            "Değerler": f"Std Dev: {curr_std:.2f}"
        })
    except: pass
    
    # --- Fibonacci (Simple Retracement based on visible range) ---
    try:
        period_high = df['High'].max()
        period_low = df['Low'].min()
        diff = period_high - period_low
        
        levels = {
            "0.0% (Tepe)": period_high,
            "23.6%": period_high - 0.236 * diff,
            "38.2%": period_high - 0.382 * diff,
            "50.0%": period_high - 0.5 * diff,
            "61.8%": period_high - 0.618 * diff,
            "100.0% (Dip)": period_low
        }
        
        # Find closest levels
        closest_support = None
        closest_resistance = None
        
        sorted_levels = sorted(levels.items(), key=lambda x: x[1])
        
        for name, level in sorted_levels:
            if level < last_close:
                closest_support = (name, level)
            elif level > last_close and closest_resistance is None:
                closest_resistance = (name, level)
                
        fib_vals = f"Destek: {closest_support[0]} ({closest_support[1]:.2f})" if closest_support else "Destek: Yok"
        fib_vals += f", Direnç: {closest_resistance[0]} ({closest_resistance[1]:.2f})" if closest_resistance else ", Direnç: Yok"
        
        signals.append({
            "İndikatör": "Fibonacci Seviyeleri",
            "Sinyal": "N/A",
            "Değerler": fib_vals
        })
    except: pass

    return pd.DataFrame(signals)


def _signals_df_to_dict(signals_df: pd.DataFrame) -> dict:
    """calculate_technical_signals çıktısını {indicator_name: row_dict} sözlüğüne çevirir."""
    out = {}
    if signals_df is None or signals_df.empty:
        return out
    for _, row in signals_df.iterrows():
        out[str(row['İndikatör'])] = {
            'sinyal': str(row.get('Sinyal', '')),
            'degerler': str(row.get('Değerler', '')),
        }
    return out


def _extract_rsi_value(degerler: str) -> float | None:
    m = re.search(r"RSI:\s*([0-9]+\.?[0-9]*)", degerler or "")
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _summarize_general(signal_map: dict) -> str:
    """AL/SAT sayısına göre genel özet üretir."""
    al = sat = bekle = 0
    for v in signal_map.values():
        s = (v.get('sinyal') or '').upper()
        if 'AL' in s and 'SAT' not in s:
            al += 1
        elif 'SAT' in s:
            sat += 1
        elif 'BEKLE' in s:
            bekle += 1
    if al == 0 and sat == 0:
        return "BEKLE"
    if al > sat * 2:
        return "GÜÇLÜ AL"
    if sat > al * 2:
        return "GÜÇLÜ SAT"
    if al > sat:
        return "AL"
    if sat > al:
        return "SAT"
    return "BEKLE"


def compute_daily_record(symbol: str) -> dict | None:
    """
    Verilen hisse için site içi hesaplamaları kullanarak bir günlük kayıt dict'i üretir.
    DB'ye yazılabilir formatta (DB kolon adlarıyla) döner.
    """
    yf_symbol = symbol if symbol.endswith('.IS') else f"{symbol}.IS"
    df = get_stock_data(yf_symbol, "1y", "1d")
    if df is None or df.empty:
        return None

    last = df.iloc[-1]

    def _f(v):
        try:
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            v = float(v)
            if pd.isna(v):
                return None
            return round(v, 4)
        except (TypeError, ValueError):
            return None

    signals_df = calculate_technical_signals(df)
    sig_map = _signals_df_to_dict(signals_df)

    rsi_value = _extract_rsi_value(sig_map.get('RSI', {}).get('degerler', ''))
    genel = _summarize_general(sig_map)

    tarih_iso = dl.today_istanbul().isoformat()
    kaydedilme = dl.now_istanbul().strftime("%Y-%m-%d %H:%M:%S")

    record = {
        'tarih': tarih_iso,
        'hisse': symbol.upper().replace('.IS', ''),
        'kapanis': _f(last.get('Close')),
        'acilis': _f(last.get('Open')),
        'yuksek': _f(last.get('High')),
        'dusuk': _f(last.get('Low')),
        'ma_sinyal': sig_map.get('Hareketli Ortalamalar (MA)', {}).get('sinyal') or '',
        'macd_sinyal': sig_map.get('MACD', {}).get('sinyal') or '',
        'rsi': rsi_value,
        'bollinger_sinyal': sig_map.get('Bollinger Bantları', {}).get('sinyal') or '',
        'stokastik': sig_map.get('Stokastik', {}).get('sinyal') or '',
        'ichimoku': sig_map.get('Ichimoku', {}).get('sinyal') or '',
        'genel_sinyal': genel,
        'kaydedilme_zamani': kaydedilme,
    }
    return record


def run_daily_capture(symbols: list[str], overwrite: bool = False, progress_cb=None) -> dict:
    """
    Verilen sembol listesi için bir kerede günlük kayıt yapar.
    overwrite=True yalnızca BUGÜNÜN kaydı için geçerlidir; geçmiş tarihler korunur (daily_log içinde).
    """
    records = []
    failed = []
    for idx, sym in enumerate(symbols):
        try:
            rec = compute_daily_record(sym)
            if rec is not None:
                records.append(rec)
            else:
                failed.append(sym)
        except Exception:
            failed.append(sym)
        if progress_cb:
            try:
                progress_cb((idx + 1) / max(1, len(symbols)), sym)
            except Exception:
                pass

    inserted, skipped = dl.insert_records(records, overwrite=overwrite)
    return {
        'inserted': inserted,
        'skipped': skipped,
        'failed': failed,
        'total': len(symbols),
    }


_auto_capture_lock = threading.Lock()


_auto_capture_state = {"date": None, "thread": None, "result": None}


def auto_capture_running() -> bool:
    """Arka plandaki otomatik kayıt işi hâlâ sürüyor mu?"""
    t = _auto_capture_state.get("thread")
    return bool(t is not None and t.is_alive())


def pop_auto_capture_result():
    """Tamamlanmış otomatik kayıt sonucunu bir kez döndürür (sonra temizler)."""
    with _auto_capture_lock:
        result = _auto_capture_state.get("result")
        _auto_capture_state["result"] = None
    return result


def maybe_run_auto_capture():
    """
    Türkiye saati 18:30'dan sonraysa ve bugün için kayıt yoksa günlük kaydı
    ARKA PLANDA başlatır. Sayfa render'ını bloklamaz, dolayısıyla kullanıcı
    dakikalarca "çalışıyor" durumunda bekleyen bir arayüz görmez.
    Aynı gün içinde yalnızca bir kez tetiklenir.
    """
    try:
        now = dl.now_istanbul()
        today_str = now.date().isoformat()

        # 18:30 ve sonrası
        if (now.hour, now.minute) < (18, 30):
            return None

        with _auto_capture_lock:
            if _auto_capture_state["date"] == today_str:
                return None
            if auto_capture_running():
                return None

            # Bugün için zaten kayıt varsa çalıştırma
            if dl.has_record_for_today():
                _auto_capture_state["date"] = today_str
                return None

            symbols = dl.get_target_symbols()
            if not symbols:
                return None

            def _worker():
                try:
                    res = run_daily_capture(symbols, overwrite=False)
                except Exception:
                    res = None
                with _auto_capture_lock:
                    _auto_capture_state["date"] = today_str
                    _auto_capture_state["result"] = res

            thread = threading.Thread(target=_worker, name="auto-daily-capture", daemon=True)
            # Cache/oturum bağlamının thread içinde de geçerli olması için
            ctx = get_script_run_ctx()
            if ctx is not None:
                add_script_run_ctx(thread, ctx)
            _auto_capture_state["thread"] = thread
            thread.start()

        return None
    except Exception:
        return None
