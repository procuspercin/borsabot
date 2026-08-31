import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import io
import json
import os
from html import escape as html_escape
import re
import subprocess
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import requests
from bs4 import BeautifulSoup
import ssl

from core import daily_log as dl

# SSL Fix
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

# --- CONFIGURATION ---
st.set_page_config(
    page_title="BorsaBot Finans",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- GOOGLE FINANCE THEME CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@400;500;700&display=swap');

    :root {
        --gf-bg: #131314;
        --gf-surface: #1e1f20;
        --gf-surface-2: #282a2c;
        --gf-border: #3c4043;
        --gf-text: #e3e3e3;
        --gf-text-dim: #9aa0a6;
        --gf-green: #81c995;
        --gf-red: #f28b82;
        --gf-blue: #8ab4f8;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Google Sans', 'Roboto', -apple-system, sans-serif;
    }

    .stApp { background-color: var(--gf-bg); color: var(--gf-text); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.2rem !important; padding-bottom: 3rem !important; max-width: 1600px; }
    [data-testid="stSidebar"] { background-color: var(--gf-bg); border-right: 1px solid var(--gf-border); }

    h1, h2, h3, h4 { color: var(--gf-text) !important; font-weight: 500 !important; letter-spacing: 0; }

    /* ---------- Üst bar (logo + arama) ---------- */
    .gf-topbar {
        display: flex; align-items: center; gap: 20px;
        padding: 4px 0 14px 0; margin-bottom: 4px;
    }
    .gf-logo { font-size: 1.55rem; font-weight: 400; color: var(--gf-text); white-space: nowrap; }
    .gf-logo b { font-weight: 700; color: var(--gf-blue); }

    /* Arama kutusu -> Google pill */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: var(--gf-surface-2) !important;
        border: 1px solid transparent !important;
        border-radius: 28px !important;
        color: var(--gf-text) !important;
        min-height: 46px;
        padding-left: 14px;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
        background-color: #303134 !important;
    }
    div[data-baseweb="popover"] ul { background-color: var(--gf-surface-2) !important; }
    div[data-baseweb="popover"] li { color: var(--gf-text) !important; }

    .stTextInput input {
        background-color: var(--gf-surface-2) !important;
        color: var(--gf-text) !important;
        border: 1px solid transparent !important;
        border-radius: 24px !important;
    }

    /* ---------- Butonlar -> Google chip/pill ---------- */
    .stButton button {
        background-color: transparent;
        color: var(--gf-text);
        border: 1px solid var(--gf-border);
        border-radius: 100px;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.35rem 1rem;
        transition: background-color .15s, border-color .15s;
    }
    .stButton button:hover {
        background-color: var(--gf-surface-2);
        border-color: #5f6368;
        color: var(--gf-text);
    }
    .stButton button:focus:not(:active) { color: var(--gf-text); border-color: var(--gf-blue); }
    .stButton button[kind="primary"] {
        background-color: var(--gf-blue); color: #062e6f; border: none; font-weight: 600;
    }
    .stButton button[kind="primary"]:hover { background-color: #a8c7fa; color: #062e6f; }

    .stDownloadButton button {
        background-color: var(--gf-surface-2); color: var(--gf-text);
        border: 1px solid var(--gf-border); border-radius: 100px; font-weight: 500;
    }

    /* ---------- Sekmeler (ABD / Avrupa ... tarzı) ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; border-bottom: 1px solid var(--gf-border); padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; border: none; border-radius: 100px;
        color: var(--gf-text-dim); font-size: 0.9rem; font-weight: 500;
        padding: 8px 18px; height: auto;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: var(--gf-surface-2); color: var(--gf-text); }
    .stTabs [aria-selected="true"] {
        background-color: #004a77 !important; color: #c2e7ff !important;
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ---------- Radio -> segment chip ---------- */
    div[role="radiogroup"] label {
        background-color: var(--gf-surface-2); border-radius: 100px;
        padding: 6px 14px; margin-right: 8px; border: 1px solid transparent;
    }
    div[role="radiogroup"] label:hover { border-color: #5f6368; }

    /* Multiselect etiketleri -> Google chip */
    span[data-baseweb="tag"] {
        background-color: #004a77 !important; color: #c2e7ff !important;
        border-radius: 100px !important; font-weight: 500;
    }
    div[data-baseweb="select"] input { color: var(--gf-text) !important; }

    /* ---------- Piyasa kartları ---------- */
    .gf-cards { display: flex; gap: 12px; overflow-x: auto; padding: 14px 2px 6px 2px; }
    .gf-cards::-webkit-scrollbar { height: 6px; }
    .gf-cards::-webkit-scrollbar-thumb { background: var(--gf-border); border-radius: 3px; }
    .gf-card {
        flex: 1 0 190px; min-width: 190px;
        background-color: var(--gf-surface);
        border: 1px solid var(--gf-border);
        border-radius: 12px;
        padding: 14px 14px 6px 14px;
    }
    .gf-card:hover { background-color: var(--gf-surface-2); }
    .gf-card-name { font-size: 0.9rem; color: var(--gf-text); font-weight: 500; margin-bottom: 8px;
                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .gf-card-price { font-size: 1.35rem; font-weight: 400; color: var(--gf-text); line-height: 1.2; }
    .gf-card-abs { font-size: 0.8rem; color: var(--gf-text-dim); margin-top: 2px; }
    .gf-card-chg { font-size: 0.95rem; font-weight: 500; margin-top: 4px; }
    .gf-up { color: var(--gf-green); }
    .gf-down { color: var(--gf-red); }
    .gf-flat { color: var(--gf-text-dim); }
    .gf-spark { margin-top: 6px; }

    /* ---------- Sol panel listeleri ---------- */
    .gf-side-title {
        font-size: 1.15rem; font-weight: 400; color: var(--gf-text);
        padding: 6px 0 10px 4px; display: flex; align-items: center; gap: 8px;
    }
    .gf-side-sub {
        font-size: 0.75rem; color: var(--gf-text-dim); text-transform: uppercase;
        letter-spacing: .6px; padding: 14px 4px 6px 4px;
    }
    .gf-row {
        display: flex; align-items: center; justify-content: space-between;
        padding: 9px 10px; border-radius: 10px; gap: 10px;
    }
    .gf-row:hover { background-color: var(--gf-surface-2); }
    .gf-row-left { display: flex; flex-direction: column; min-width: 0; }
    .gf-row-sym { font-size: 0.88rem; color: var(--gf-text); font-weight: 500; }
    .gf-row-name { font-size: 0.72rem; color: var(--gf-text-dim); white-space: nowrap;
                   overflow: hidden; text-overflow: ellipsis; max-width: 110px; }
    .gf-row-right { display: flex; align-items: center; gap: 10px; }
    .gf-row-price { font-size: 0.85rem; color: var(--gf-text); text-align: right; }
    .gf-row-chg { font-size: 0.78rem; font-weight: 500; min-width: 58px; text-align: right; }
    .gf-empty { color: var(--gf-text-dim); font-size: 0.82rem; padding: 8px 4px; }

    /* ---------- Haber / özet ---------- */
    .gf-panel {
        background-color: var(--gf-surface); border: 1px solid var(--gf-border);
        border-radius: 16px; padding: 18px 20px; margin-bottom: 14px;
    }
    .gf-panel h4 { margin: 0 0 10px 0; font-size: 1.05rem; font-weight: 500; }
    .gf-news { padding: 12px 0; border-bottom: 1px solid var(--gf-border); }
    .gf-news:last-child { border-bottom: none; }
    .gf-news a {
        font-size: 0.9rem; color: var(--gf-text); text-decoration: none;
        font-weight: 500; display: block; line-height: 1.35;
    }
    .gf-news a:hover { color: var(--gf-blue); text-decoration: underline; }
    .gf-news-time { font-size: 0.72rem; color: var(--gf-text-dim); margin-top: 5px; }
    .gf-source { font-size: 0.72rem; color: var(--gf-text-dim); }

    /* ---------- Sinyal rozetleri ---------- */
    .gf-badge {
        display: inline-block; padding: 3px 10px; border-radius: 100px;
        font-size: 0.75rem; font-weight: 600;
    }
    .gf-badge-buy { background: rgba(129,201,149,.15); color: var(--gf-green); }
    .gf-badge-sell { background: rgba(242,139,130,.15); color: var(--gf-red); }
    .gf-badge-wait { background: rgba(138,180,248,.15); color: var(--gf-blue); }

    /* ---------- Tablolar ---------- */
    [data-testid="stDataFrame"] {
        background-color: var(--gf-surface); border: 1px solid var(--gf-border);
        border-radius: 12px; overflow: hidden;
    }
    [data-testid="stDataFrame"] th { background-color: var(--gf-surface-2) !important; color: var(--gf-text-dim) !important; }
    [data-testid="stDataFrame"] td { color: var(--gf-text) !important; border-bottom: 1px solid var(--gf-border) !important; }

    [data-testid="stMetricValue"] { color: var(--gf-text); font-weight: 400; }
    [data-testid="stMetricLabel"] { color: var(--gf-text-dim); }

    hr { border-color: var(--gf-border); }
    .gf-divider { height: 1px; background: var(--gf-border); margin: 18px 0; border: none; }

    /* Sol paneldeki liste satırları: sola yaslı, çerçevesiz */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child .stButton button,
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child .stButton button {
        justify-content: flex-start; text-align: left;
        border-color: transparent; border-radius: 10px; padding: 8px 12px;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child .stButton button:hover,
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child .stButton button:hover {
        background-color: var(--gf-surface-2); border-color: transparent;
    }

    /* ---------- AI sohbet ---------- */
    [data-testid="stChatMessage"] {
        background-color: transparent; padding: 6px 0;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.86rem; line-height: 1.5;
    }
    [data-testid="stChatInput"] {
        background-color: var(--gf-surface-2); border-radius: 24px; border: 1px solid transparent;
    }
    [data-testid="stChatInput"] textarea { color: var(--gf-text) !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: var(--gf-text-dim) !important; }

    /* Popup penceresi */
    div[role="dialog"] {
        background-color: var(--gf-surface) !important;
        border: 1px solid var(--gf-border) !important;
        border-radius: 20px !important;
    }

    [data-testid="stExpander"] {
        background-color: var(--gf-surface); border: 1px solid var(--gf-border); border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- FULL STOCK LIST ---
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

# --- PİYASA KATEGORİLERİ (Google Finance sekmeleri karşılığı) ---
# Google'daki "ABD / Avrupa / Asya / Latin Amerika / Para birimleri / Kripto / Vadeli"
# yerine bizim piyasamıza uygun sekmeler.
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

# Sol paneldeki "Hisse senedi sektörleri" karşılığı (BIST sektör endeksleri)
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

# Üst şeritteki mini özet (her sekmede sabit)
TICKER_STRIP = [
    ("XU100.IS", "BIST 100"),
    ("USDTRY=X", "Dolar"),
    ("EURTRY=X", "Euro"),
    ("GC=F", "Ons Altın"),
    ("BZ=F", "Brent"),
    ("BTC-USD", "Bitcoin"),
]

# --- STATE MANAGEMENT ---
if 'view' not in st.session_state:
    st.session_state.view = 'home'
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'THYAO.IS'
if 'last_viewed' not in st.session_state:
    st.session_state.last_viewed = []
if 'popup_symbol' not in st.session_state:
    st.session_state.popup_symbol = None
if 'pending_answer' not in st.session_state:
    st.session_state.pending_answer = None
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["THYAO.IS", "ASELS.IS", "SISE.IS", "GARAN.IS"]

def remember_symbol(symbol):
    """Seçili sembolü ve 'son bakılanlar' listesini günceller (rerun tetiklemez)."""
    st.session_state.selected_symbol = symbol
    lv = st.session_state.last_viewed
    if symbol in lv:
        lv.remove(symbol)
    lv.insert(0, symbol)
    del lv[12:]


def go_to_detail(symbol):
    remember_symbol(symbol)
    st.session_state.view = 'detail'
    st.rerun()


def handle_global_search():
    """
    Arama kutusu callback'i. Değeri hemen sıfırlar; aksi halde her yeniden
    çalıştırmada aynı sembol tekrar seçili sayılıp sonsuz rerun döngüsü oluşur.
    """
    sym = st.session_state.get("global_search")
    st.session_state.global_search = None
    if sym:
        remember_symbol(sym)
        st.session_state.view = 'detail'


def handle_watchlist_add():
    """İzleme listesine ekleme callback'i; seçim tüketilip sıfırlanır."""
    sym = st.session_state.get("wl_add")
    st.session_state.wl_add = None
    if sym and sym not in st.session_state.watchlist:
        st.session_state.watchlist.insert(0, sym)

def go_to_home():
    st.session_state.view = 'home'
    st.rerun()

def go_to_bulk():
    st.session_state.view = 'bulk'
    st.rerun()

def go_to_daily_log():
    st.session_state.view = 'daily_log'
    st.rerun()

def go_to_forecast(symbol: str | None = None):
    if symbol:
        code = symbol.replace('.IS', '')
        st.session_state.selected_symbol = symbol
        st.session_state.fc_last = code
        # Açılır listeyi de aynı hisseye çek; aksi halde liste eski hisseyi
        # gösterirken rapor yeni hisseye ait olur.
        if code in ml_supported_tickers():
            st.session_state.fc_symbol = code
    st.session_state.view = 'forecast'
    st.rerun()

def toggle_watch(symbol):
    wl = st.session_state.watchlist
    if symbol in wl:
        wl.remove(symbol)
    else:
        wl.insert(0, symbol)
    st.session_state.watchlist = wl

# --- DATA FETCHING ---

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


@st.cache_data(ttl=120, show_spinner=False)
def get_quotes(tickers: tuple, period: str = "1mo", interval: str = "1d") -> dict:
    """
    Verilen semboller için {ticker: {price, change, pct, spark}} döndürür.
    spark: mini grafik için son kapanış listesi.
    """
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return {}
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
        pct = (change / prev * 100) if prev else 0.0
        out[t] = {
            "price": price,
            "change": change,
            "pct": pct,
            "spark": [float(v) for v in close.tail(40).tolist()],
        }
    return out


@st.cache_data(ttl=300, show_spinner=False)
def get_market_movers(limit: int = 40) -> pd.DataFrame:
    """En çok yükselen / düşen hisseler."""
    tickers = tuple(s for s in BIST100_SYMBOLS if not s.startswith("XU"))[:limit]
    quotes = get_quotes(tickers)
    rows = []
    for t, q in quotes.items():
        rows.append({"Sembol": t.replace(".IS", ""), "Fiyat": q["price"], "Değişim %": q["pct"]})
    if not rows:
        return pd.DataFrame(columns=["Sembol", "Fiyat", "Değişim %"])
    return pd.DataFrame(rows).sort_values("Değişim %", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
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


def get_stock_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


# --- GÖRSEL YARDIMCILAR ---

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

# --- COMPONENTS ---

def render_topbar(show_search: bool = True):
    """Google Finance üst çubuğu: logo + arama + hızlı gezinme."""
    c_logo, c_search, c_nav1, c_nav2, c_nav3 = st.columns([2.1, 4.4, 1.25, 1.25, 1.25])
    with c_logo:
        st.markdown(
            '<div class="gf-topbar"><span class="gf-logo">Borsa<b>Bot</b> Finans</span></div>',
            unsafe_allow_html=True,
        )
    with c_search:
        if show_search:
            st.selectbox(
                "Ara",
                BIST100_SYMBOLS,
                index=None,
                placeholder="Hisse senedi, endeks veya sembol arayın",
                label_visibility="collapsed",
                key="global_search",
                on_change=handle_global_search,
            )
    with c_nav1:
        if st.button("Toplu Analiz", use_container_width=True, key="nav_bulk"):
            go_to_bulk()
    with c_nav2:
        if st.button("Günlük Kayıt", use_container_width=True, key="nav_log"):
            go_to_daily_log()
    with c_nav3:
        if st.button("ML Tahmin", use_container_width=True, key="nav_forecast"):
            go_to_forecast()


def render_ticker_strip():
    """Üstte ince piyasa şeridi (BIST, dolar, altın, brent, bitcoin)."""
    quotes = get_quotes(tuple(t for t, _ in TICKER_STRIP))
    if not quotes:
        return
    html = '<div class="gf-cards" style="padding-top:2px;padding-bottom:2px;">'
    for tkr, name in TICKER_STRIP:
        q = quotes.get(tkr)
        if not q:
            continue
        cls = trend_class(q["pct"])
        sign = "+" if q["pct"] >= 0 else ""
        html += (
            f'<div class="gf-card" style="flex:1 0 150px;min-width:150px;padding:10px 12px;">'
            f'<div class="gf-card-name" style="font-size:.75rem;color:var(--gf-text-dim);margin-bottom:4px;">{name}</div>'
            f'<div style="display:flex;align-items:baseline;gap:8px;">'
            f'<span style="font-size:1rem;font-weight:500;">{fmt_price(q["price"])}</span>'
            f'<span class="{cls}" style="font-size:.8rem;">{sign}{q["pct"]:.2f}%</span>'
            f'</div></div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def quote_button(sym: str, q: dict | None, label: str | None = None, key_prefix: str = "q"):
    """Tıklanabilir liste satırı (Google'ın izleme listesi satırı karşılığı)."""
    show = label or sym.replace(".IS", "")
    if q:
        pct = q["pct"]
        sign = "+" if pct >= 0 else ""
        color = "green" if pct >= 0 else "red"
        text = f"**{show}** &nbsp; {fmt_price(q['price'])} &nbsp; :{color}[{sign}{pct:.2f}%]"
    else:
        text = f"**{show}** &nbsp; :gray[veri yok]"
    if st.button(text, key=f"{key_prefix}_{sym}", use_container_width=True):
        st.session_state.selected_symbol = sym
        st.session_state.popup_symbol = sym
        st.rerun()


def render_side_lists():
    """Sol panel: İzleme listesi, son bakılanlar, sektör endeksleri."""
    st.markdown('<div class="gf-side-title">☰ Listeler</div>', unsafe_allow_html=True)

    # --- İzleme listesi ---
    st.markdown('<div class="gf-side-sub">İzleme listesi</div>', unsafe_allow_html=True)
    wl = st.session_state.watchlist
    if wl:
        quotes = get_quotes(tuple(wl))
        for sym in wl:
            quote_button(sym, quotes.get(sym), key_prefix="wl")
    else:
        st.markdown('<div class="gf-empty">Bu liste boş.</div>', unsafe_allow_html=True)

    st.selectbox(
        "İzlemeye ekle", BIST100_SYMBOLS, index=None,
        placeholder="+ İzleme listesine ekle", label_visibility="collapsed", key="wl_add",
        on_change=handle_watchlist_add,
    )

    # --- Son bakılanlar ---
    if st.session_state.last_viewed:
        st.markdown('<div class="gf-side-sub">Son bakılanlar</div>', unsafe_allow_html=True)
        lv = st.session_state.last_viewed[:6]
        lv_quotes = get_quotes(tuple(lv))
        for sym in lv:
            quote_button(sym, lv_quotes.get(sym), key_prefix="lv")

    # --- Sektör endeksleri ---
    st.markdown('<div class="gf-side-sub">Hisse senedi sektörleri</div>', unsafe_allow_html=True)
    sec_quotes = get_quotes(tuple(t for t, _ in SECTOR_INDICES))
    rows = ""
    for tkr, name in SECTOR_INDICES:
        q = sec_quotes.get(tkr)
        if not q:
            continue
        rows += quote_row_html(tkr.replace(".IS", ""), name, q)
    if rows:
        st.markdown(rows, unsafe_allow_html=True)
    else:
        st.markdown('<div class="gf-empty">Sektör verisi alınamadı.</div>', unsafe_allow_html=True)


def render_market_tabs():
    """Google Finance'in ABD/Avrupa/Asya sekmeleri karşılığı."""
    tabs = st.tabs(list(MARKET_TABS.keys()))
    for tab, (tab_name, items) in zip(tabs, MARKET_TABS.items()):
        with tab:
            quotes = get_quotes(tuple(t for t, _ in items))
            if not quotes:
                st.markdown('<div class="gf-empty">Veri alınamadı.</div>', unsafe_allow_html=True)
                continue
            html = '<div class="gf-cards">'
            for tkr, name in items:
                q = quotes.get(tkr)
                if q:
                    html += market_card_html(name, q)
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)


def render_movers():
    """Yükselen / düşen hisseler (Google'ın 'En çok yükselenler' bölümü)."""
    movers = get_market_movers()
    if movers.empty:
        st.markdown('<div class="gf-empty">Hisse verisi alınamadı.</div>', unsafe_allow_html=True)
        return

    # Az sayıda veri geldiğinde aynı hissenin hem yükselenlerde hem düşenlerde
    # görünmemesi için liste uzunluğunu ikiye böl.
    n_side = max(1, min(6, len(movers) // 2))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="gf-side-sub">En çok yükselenler</div>', unsafe_allow_html=True)
        html = ""
        for _, r in movers.head(n_side).iterrows():
            html += quote_row_html(r["Sembol"], "BIST", {"price": r["Fiyat"], "pct": r["Değişim %"]})
        st.markdown(html, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="gf-side-sub">En çok düşenler</div>', unsafe_allow_html=True)
        html = ""
        for _, r in movers.tail(n_side).sort_values("Değişim %").iterrows():
            html += quote_row_html(r["Sembol"], "BIST", {"price": r["Fiyat"], "pct": r["Değişim %"]})
        st.markdown(html, unsafe_allow_html=True)

    with st.expander("Tüm hisseleri tablo olarak gör"):
        st.dataframe(
            movers.style.format({"Fiyat": "{:.2f}", "Değişim %": "{:+.2f}%"})
                  .map(lambda v: "color:#81c995" if isinstance(v, float) and v > 0
                       else ("color:#f28b82" if isinstance(v, float) and v < 0 else ""),
                       subset=["Değişim %"]),
            use_container_width=True, hide_index=True, height=420,
        )


def render_news(limit: int = 12):
    news = get_bloomberg_news()
    if not news:
        st.markdown('<div class="gf-empty">Haber akışı alınamadı.</div>', unsafe_allow_html=True)
        return
    html = ""
    for item in news[:limit]:
        html += (
            f'<div class="gf-news">'
            f'<a href="{html_escape(item["link"], quote=True)}" target="_blank" rel="noopener">'
            f'{html_escape(item["title"])}</a>'
            f'<div class="gf-news-time">Bloomberg HT · {html_escape(item["published"][5:16])}</div></div>'
        )
    st.markdown(html, unsafe_allow_html=True)


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


@st.cache_data(ttl=300, show_spinner=False)
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


# --- GEMINI ASİSTANI ---

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



# Kullanıcı "aselsan nasıl" diye sorduğunda hangi sembolün verisini çekeceğimiz
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


def get_gemini_key() -> str:
    key = st.session_state.get("gemini_key", "") or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            key = ""
    return (key or "").strip()


def ask_gemini(history: list, context_text: str):
    """Gemini'ye teknik veri bağlamıyla soru sorar. (cevap, hata) döndürür."""
    key = get_gemini_key()
    if not key:
        return None, "Gemini API anahtarı tanımlı değil."

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


def _chat_key(symbol: str) -> str:
    return f"chat_{symbol}"


def send_to_assistant(symbol: str, question: str):
    """Soruyu sohbete ekler; yanıt bir sonraki render'da üretilir."""
    k = _chat_key(symbol)
    st.session_state.setdefault(k, [])
    st.session_state[k].append({"role": "user", "content": question})
    st.session_state.pending_answer = symbol


def render_ai_panel():
    """Sağ panel: seçili hisse için Gemini destekli teknik analiz sohbeti."""
    symbol = st.session_state.selected_symbol
    short = symbol.replace(".IS", "")

    st.markdown(
        f'<div class="gf-side-title">✨ Teknik Analiz Asistanı</div>'
        f'<div class="gf-source" style="padding-left:4px;">Konu: <b>{short}</b> · Gemini</div>',
        unsafe_allow_html=True,
    )

    if not get_gemini_key():
        st.markdown(
            '<div class="gf-panel" style="padding:14px 16px;">'
            '<div style="font-size:.88rem;">Asistanı kullanmak için Gemini API anahtarı gerekli.</div>'
            '<div class="gf-source" style="margin-top:6px;">aistudio.google.com/apikey adresinden '
            'ücretsiz alabilirsin. Anahtarı <code>.streamlit/secrets.toml</code> içine '
            '<code>GEMINI_API_KEY="..."</code> olarak yazabilir ya da aşağıya yapıştırabilirsin '
            '(sadece bu oturumda saklanır).</div></div>',
            unsafe_allow_html=True,
        )
        entered = st.text_input(
            "Gemini API anahtarı", type="password", key="gemini_key_input",
            placeholder="API anahtarını yapıştır", label_visibility="collapsed",
        )
        if entered:
            st.session_state.gemini_key = entered
            st.rerun()
        return

    ctx = get_analysis_context(symbol)
    if ctx is None:
        st.markdown('<div class="gf-empty">Bu sembol için veri alınamadı.</div>', unsafe_allow_html=True)
        return

    k = _chat_key(symbol)
    history = st.session_state.setdefault(k, [])

    # Hazır sorular (Google'ın öneri kartları yerine)
    if not history:
        st.markdown('<div class="gf-side-sub">Hazır sorular</div>', unsafe_allow_html=True)
        presets = [
            f"{short} teknik olarak nasıl görünüyor?",
            "RSI ve MACD şu an ne gösteriyor?",
            "Hangi seviyeler destek, hangileri direnç?",
            "Geri çekilmede hangi bandlar izlenir?",
        ]
        for i, p in enumerate(presets):
            if st.button(p, key=f"preset_{symbol}_{i}", use_container_width=True):
                send_to_assistant(symbol, p)
                st.rerun()

    # Sohbet geçmişi
    chat_box = st.container(height=360 if history else 120)
    with chat_box:
        if not history:
            st.markdown(
                '<div class="gf-empty">Bu hisseyle ilgili indikatör verileri asistana otomatik '
                'aktarılıyor. Bir soru sor ya da yukarıdan hazır bir soruyu seç.</div>',
                unsafe_allow_html=True,
            )
        for m in history:
            with st.chat_message("user" if m["role"] == "user" else "assistant",
                                 avatar="🧑" if m["role"] == "user" else "✨"):
                st.markdown(m["content"])

        # Bekleyen soru varsa cevabı üret
        if st.session_state.get("pending_answer") == symbol:
            with st.chat_message("assistant", avatar="✨"):
                with st.spinner("Teknik veriler yorumlanıyor..."):
                    context_text = ctx["text"]
                    son_soru = next(
                        (m["content"] for m in reversed(history) if m["role"] == "user"), ""
                    )
                    for extra_sym in detect_symbols(son_soru, exclude=symbol)[:2]:
                        extra_ctx = get_analysis_context(extra_sym)
                        if extra_ctx:
                            context_text += "\n\n" + extra_ctx["text"]
                    answer, err = ask_gemini(history, context_text)
            st.session_state.pending_answer = None
            if answer:
                history.append({"role": "assistant", "content": answer})
            else:
                history.append({"role": "assistant", "content": f"⚠️ {err}"})
            st.rerun()

    prompt = st.chat_input(f"{short} hakkında teknik bir soru sor...", key=f"chat_input_{symbol}")
    if prompt:
        send_to_assistant(symbol, prompt)
        st.rerun()

    st.markdown(
        '<div class="gf-source" style="padding:6px 2px;">Başka bir hisseyi sorarsan '
        'onun teknik verisini de getiririm. Yatırım tavsiyesi değildir.</div>',
        unsafe_allow_html=True,
    )
    if history and st.button("Sohbeti temizle", key=f"clear_{symbol}", use_container_width=True):
        st.session_state[k] = []
        st.rerun()


# --- HİSSE POPUP (teknik özet) ---

@st.dialog("Teknik özet", width="large")
def stock_popup(symbol: str):
    short = symbol.replace(".IS", "")
    ctx = get_analysis_context(symbol)

    if ctx is None:
        st.markdown('<div class="gf-empty">Bu sembol için veri alınamadı.</div>', unsafe_allow_html=True)
        return

    q = ctx["quote"]
    cls = trend_class(q["pct"])
    sign = "+" if q["pct"] >= 0 else ""

    st.markdown(
        f'<div style="display:flex;align-items:flex-end;justify-content:space-between;gap:16px;">'
        f'<div><div style="font-size:1.25rem;font-weight:500;">{short}'
        f'<span class="gf-source" style="margin-left:8px;">BIST · {symbol}</span></div>'
        f'<div style="display:flex;align-items:baseline;gap:10px;margin-top:6px;">'
        f'<span style="font-size:1.9rem;font-weight:400;">{fmt_price(q["price"])}</span>'
        f'<span class="{cls}" style="font-size:.95rem;font-weight:500;">'
        f'{sign}{fmt_price(q["change"])} ({sign}{q["pct"]:.2f}%) {trend_arrow(q["pct"])}</span></div></div>'
        f'<div style="text-align:right;"><div class="gf-source">Genel görünüm</div>'
        f'<div style="margin-top:6px;">{signal_badge(ctx["genel"])}</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="margin-top:6px;">{sparkline_svg(q["spark"], q["pct"], width=520, height=90)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="gf-side-sub">İndikatör sinyalleri</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    rows = list(ctx["signals"].iterrows())
    half = (len(rows) + 1) // 2
    for col, chunk in ((left, rows[:half]), (right, rows[half:])):
        html = ""
        for _, r in chunk:
            sig = str(r["Sinyal"])
            badge = signal_badge(sig) if sig not in ("", "N/A") else '<span class="gf-source">—</span>'
            html += (
                f'<div class="gf-row"><div class="gf-row-left">'
                f'<span class="gf-row-sym">{r["İndikatör"]}</span>'
                f'<span class="gf-row-name" style="max-width:220px;">{r["Değerler"]}</span></div>'
                f'<div class="gf-row-right">{badge}</div></div>'
            )
        col.markdown(html, unsafe_allow_html=True)

    st.markdown('<hr class="gf-divider">', unsafe_allow_html=True)

    ml_ok = short in ml_supported_tickers()
    b1, b2, b3, b4 = st.columns([1.25, 1.25, 1.1, 1.1]) if ml_ok else (*st.columns([1.3, 1.3, 1.2]), None)
    with b1:
        if st.button("Detaylı grafiği aç", use_container_width=True, key="pop_detail", type="primary"):
            st.session_state.popup_symbol = None
            go_to_detail(symbol)
    with b2:
        in_wl = symbol in st.session_state.watchlist
        if st.button("★ İzlemeden çıkar" if in_wl else "☆ İzlemeye ekle",
                     use_container_width=True, key="pop_watch"):
            toggle_watch(symbol)
            st.session_state.popup_symbol = None
            st.rerun()
    with b3:
        if st.button("✨ Asistana sor", use_container_width=True, key="pop_ai"):
            st.session_state.selected_symbol = symbol
            send_to_assistant(symbol, f"{short} teknik olarak nasıl görünüyor?")
            st.session_state.popup_symbol = None
            st.rerun()
    if b4 is not None:
        with b4:
            if st.button("🤖 ML tahmini", use_container_width=True, key="pop_ml"):
                st.session_state.popup_symbol = None
                go_to_forecast(symbol)

    st.caption("Sinyaller indikatörlerin mekanik çıktısıdır, yatırım tavsiyesi değildir.")


def maybe_show_popup():
    """popup_symbol ayarlanmışsa teknik özet penceresini açar."""
    sym = st.session_state.get("popup_symbol")
    if sym:
        # Aynı çalıştırmada bayrağı temizle: pencere içindeki her etkileşim pencereyi kapatır.
        st.session_state.popup_symbol = None
        stock_popup(sym)



# --- ML TAHMİN (stock_forecaster köprüsü) ---

FORECASTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_forecaster")
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


@st.cache_data(ttl=3600, show_spinner=False)
def ml_supported_tickers() -> list:
    if not forecaster_available():
        return []
    data, _ = _run_forecaster(["--list"], timeout=60)
    return data.get("tickers", []) if data else []


@st.cache_data(ttl=1800, show_spinner=False)
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


def _forecast_chart(data: dict):
    """Geçmiş fiyat + vadelere göre beklenti aralığı (fan grafiği)."""
    hist = pd.DataFrame(data["history"])
    if hist.empty:
        return None
    hist["date"] = pd.to_datetime(hist["date"])
    last_date = hist["date"].iloc[-1]
    close = data["close"]

    xs, med, low, high = [last_date], [close], [close], [close]
    for f in data["forecasts"]:
        xs.append(last_date + pd.tseries.offsets.BDay(f["horizon"]))
        med.append(f["median_price"])
        low.append(f["low_price"])
        high.append(f["high_price"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["close"], name="Geçmiş fiyat",
        line=dict(color="#8ab4f8", width=1.6),
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=high, name="P75", mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=low, name="Olası aralık (P25–P75)", mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(129,201,149,0.13)",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=med, name="Tipik beklenti (medyan)",
        line=dict(color="#81c995", width=2, dash="dot"),
        mode="lines+markers", marker=dict(size=6),
    ))
    fig.add_vline(x=last_date, line_dash="dash", line_color="#5f6368", line_width=1)
    fig.update_layout(
        height=420, template="plotly_dark",
        paper_bgcolor="#131314", plot_bgcolor="#131314",
        margin=dict(l=0, r=40, t=10, b=20),
        xaxis=dict(showgrid=True, gridcolor="#2c2d2f"),
        yaxis=dict(showgrid=True, gridcolor="#2c2d2f", side="right"),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    return fig


def render_forecast():
    render_topbar(show_search=False)

    b1, b2 = st.columns([0.7, 9])
    with b1:
        if st.button("← Geri", key="fc_back"):
            go_to_home()
    with b2:
        st.markdown("### ML fiyat tahmini")

    if not forecaster_available():
        st.error("`stock_forecaster` klasörü bulunamadı. Model dosyaları projede olmalı.")
        return

    tickers = ml_supported_tickers()
    if not tickers:
        st.error("Model için eğitilmiş hisse listesi okunamadı.")
        return

    st.caption(
        "RandomForest yön modelleri (10/30/60/120/180 işlem günü) + geçmiş olasılık kalibrasyonu. "
        "Model 2010–2026 arası veriyle eğitildi ve yalnızca aşağıdaki hisseleri kapsıyor."
    )

    c_sym, c_run = st.columns([3, 1])
    with c_sym:
        default = st.session_state.selected_symbol.replace(".IS", "")
        idx = tickers.index(default) if default in tickers else tickers.index("THYAO") if "THYAO" in tickers else 0
        ticker = st.selectbox("Hisse", tickers, index=idx, key="fc_symbol")
    with c_run:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        run = st.button("Tahmin üret", type="primary", use_container_width=True, key="fc_run")

    if run:
        st.session_state.fc_last = ticker

    target = st.session_state.get("fc_last")
    if not target:
        st.info("Bir hisse seçip **Tahmin üret** butonuna bas.")
        return

    with st.spinner(f"{target} için modeller çalıştırılıyor..."):
        data, err = ml_forecast(target)

    if err:
        st.error(f"Tahmin üretilemedi: {err}")
        return

    # --- Başlık: son veri tarihi / kapanış / piyasa rejimi ---
    last_date = pd.to_datetime(data["date"])
    gecen_gun = (pd.Timestamp(dl.today_istanbul()) - last_date).days
    regime_tr = {"BULL": "Yükseliş eğilimi", "BEAR": "Düşüş eğilimi", "NEUTRAL": "Nötr"}

    h1, h2, h3 = st.columns(3)
    h1.metric("Son veri tarihi", last_date.strftime("%d.%m.%Y"))
    h2.metric("Son kapanış", f"{fmt_price(data['close'])} TL")
    h3.metric("Piyasa rejimi (BIST 100)", regime_tr.get(data["market_regime"], data["market_regime"]))

    if gecen_gun > 5:
        st.warning(
            f"Modelin kullandığı fiyat verisi {gecen_gun} gün eski. "
            "Aşağıdaki **Model verisini güncelle** butonuyla yenileyebilirsin."
        )

    # --- Vade kartları ---
    st.markdown('<div class="gf-side-sub">Vadelere göre beklenti</div>', unsafe_allow_html=True)
    cards = '<div class="gf-cards">'
    for f in data["forecasts"]:
        cards += _forecast_card_html(f)
    cards += "</div>"
    st.markdown(cards, unsafe_allow_html=True)

    # --- Grafik ---
    fig = _forecast_chart(data)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    # --- Modelin kullandığı göstergeler ---
    with st.expander("Modelin girdi olarak kullandığı güncel göstergeler"):
        f = data["features"]
        pretty = {
            "rsi_14": ("RSI (14)", "{:.1f}"),
            "macd": ("MACD", "{:.3f}"),
            "macd_signal": ("MACD sinyal", "{:.3f}"),
            "stoch_k": ("Stokastik %K", "{:.1f}"),
            "return_20d": ("Son 20 gün getirisi", "{:+.2%}"),
            "return_60d": ("Son 60 gün getirisi", "{:+.2%}"),
            "close_sma_50_ratio": ("50 günlük ortalamaya uzaklık", "{:+.2%}"),
            "close_sma_200_ratio": ("200 günlük ortalamaya uzaklık", "{:+.2%}"),
            "volume_ratio_20": ("Hacim / 20 gün ort.", "{:.2f}x"),
            "volatility_20d": ("20 günlük oynaklık", "{:.2%}"),
            "position_52w": ("52 hafta bandındaki konum", "{:.2f}"),
            "market_return_20d": ("BIST 100 son 20 gün", "{:+.2%}"),
            "market_return_60d": ("BIST 100 son 60 gün", "{:+.2%}"),
            "market_sma_50_distance": ("BIST 100 – 50g ortalama", "{:+.2%}"),
            "market_sma_200_distance": ("BIST 100 – 200g ortalama", "{:+.2%}"),
        }
        rows = []
        for key, (label, fmt) in pretty.items():
            if key in f:
                rows.append({"Gösterge": label, "Değer": fmt.format(f[key])})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Model başarısı (dürüst tablo) ---
    with st.expander("Modelin geçmiş başarısı (bunu mutlaka oku)"):
        ev = pd.DataFrame(data["evaluation"])
        if not ev.empty:
            ev = ev.rename(columns={
                "horizon": "Vade (gün)", "auc": "OOS AUC", "accuracy": "Doğruluk",
                "worst_year_auc": "En kötü yıl AUC", "best_year_auc": "En iyi yıl AUC",
                "years_above_050": "0.50 üstü yıl sayısı",
            })
            st.dataframe(
                ev.style.format({
                    "OOS AUC": "{:.3f}", "Doğruluk": "{:.1%}",
                    "En kötü yıl AUC": "{:.3f}", "En iyi yıl AUC": "{:.3f}",
                }),
                use_container_width=True, hide_index=True,
            )
        st.markdown(
            "AUC 0.50 = yazı tura. Modelin gerçek başarısı 0.51–0.57 aralığında, yani rastgeleden "
            "yalnızca bir miktar iyi ve bazı yıllarda 0.50'nin altına düşüyor. Kartlardaki "
            "\"tipik beklenti\", benzer olasılık üretilen geçmiş örneklerin **medyan** getirisidir; "
            "garanti değil, geçmiş dağılımın özetidir."
        )

    st.caption(
        "Bu tahminler geçmiş fiyat verisi ve teknik göstergelerden üretilen istatistiksel "
        "beklentilerdir. AL/SAT kararı üretmez, yatırım tavsiyesi değildir."
    )

    # --- Veri güncelleme ---
    st.markdown('<hr class="gf-divider">', unsafe_allow_html=True)
    u1, u2 = st.columns([1.2, 3])
    with u1:
        if st.button("Model verisini güncelle", key="fc_update", use_container_width=True):
            with st.spinner("Hisse ve BIST 100 verileri indiriliyor (birkaç dakika sürebilir)..."):
                logs = update_forecaster_data()
            ml_forecast.clear()
            ml_supported_tickers.clear()
            st.success("Veri güncelleme tamamlandı: " + " | ".join(logs))
            st.rerun()
    with u2:
        st.caption(
            "28 hisse + BIST 100 için 2010'dan bugüne fiyat verisi yeniden indirilir. "
            "Modeller yeniden eğitilmez, sadece girdi verisi tazelenir."
        )


def render_home_forecast_entry():
    """Ana sayfanın altındaki ekstra seçenek: ML fiyat tahmini girişi."""
    tickers = ml_supported_tickers()
    st.markdown("#### Makine öğrenmesi tahmini")
    if not tickers:
        st.markdown(
            '<div class="gf-empty">Tahmin modeli bulunamadı (stock_forecaster klasörü gerekli).</div>',
            unsafe_allow_html=True,
        )
        return

    st.caption(
        "Hisseyi seç, RandomForest yön modelleri 10 / 30 / 60 / 120 / 180 işlem günü için "
        "beklenen fiyat aralığını üretsin. Yatırım tavsiyesi değildir."
    )
    c1, c2 = st.columns([3, 1.2])
    with c1:
        default = st.session_state.selected_symbol.replace(".IS", "")
        idx = tickers.index(default) if default in tickers else 0
        pick = st.selectbox("Tahmin için hisse", tickers, index=idx,
                            label_visibility="collapsed", key="home_fc_symbol")
    with c2:
        if st.button("Tahmini göster", type="primary", use_container_width=True, key="home_fc_run"):
            go_to_forecast(f"{pick}.IS")


def render_home():
    render_topbar()
    maybe_show_popup()
    render_ticker_strip()

    col_side, col_main, col_right = st.columns([1.25, 3.3, 1.45], gap="medium")

    with col_side:
        render_side_lists()

    with col_main:
        render_market_tabs()
        st.markdown('<hr class="gf-divider">', unsafe_allow_html=True)
        st.markdown("#### Piyasa özeti")
        render_movers()
        st.markdown('<hr class="gf-divider">', unsafe_allow_html=True)
        st.markdown("#### Bugün piyasalarda neler oluyor?")
        render_news(10)
        st.markdown('<hr class="gf-divider">', unsafe_allow_html=True)
        render_home_forecast_entry()

    with col_right:
        render_ai_panel()

def render_detail():
    symbol = st.session_state.selected_symbol

    render_topbar()

    # Başlık: fiyat, değişim ve izleme listesi butonu (Google Finance hisse sayfası tarzı)
    c_back, c_title, c_watch = st.columns([0.7, 7, 2])
    with c_back:
        if st.button("← Geri", key="detail_back"):
            go_to_home()
    with c_title:
        q = get_quotes((symbol,)).get(symbol)
        if q:
            cls = trend_class(q["pct"])
            sign = "+" if q["pct"] >= 0 else ""
            st.markdown(
                f'<div style="padding:4px 0 2px 0;">'
                f'<div style="font-size:1.5rem;font-weight:500;">{symbol.replace(".IS", "")}'
                f'<span class="gf-source" style="margin-left:10px;">BIST · {symbol}</span></div>'
                f'<div style="display:flex;align-items:baseline;gap:12px;margin-top:6px;">'
                f'<span style="font-size:2rem;font-weight:400;">{fmt_price(q["price"])}</span>'
                f'<span class="{cls}" style="font-size:1rem;font-weight:500;">'
                f'{sign}{fmt_price(q["change"])} ({sign}{q["pct"]:.2f}%) {trend_arrow(q["pct"])}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<div style="font-size:1.5rem;font-weight:500;">{symbol}</div>',
                        unsafe_allow_html=True)
    with c_watch:
        in_wl = symbol in st.session_state.watchlist
        if st.button("★ İzlemeden çıkar" if in_wl else "☆ İzlemeye ekle",
                     use_container_width=True, key="detail_watch"):
            toggle_watch(symbol)
            st.rerun()

    # Chart Controls
    # Chart Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 4])
    with col_ctrl1:
        # Default index set to '1d' (index 5) initially
        interval = st.selectbox("Zaman Dilimi", ["1m", "5m", "15m", "1h", "4h", "1d", "1wk", "1mo"], index=5)
    with col_ctrl2:
        # Default index set to '1y' (index 5) initially
        period = st.selectbox("Periyot", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=5)
    with col_ctrl3:
        indicators = st.multiselect("İndikatörler", ["MA20", "MA50", "MA200", "RSI", "MACD", "Bollinger", "Stoch", "Ichimoku", "CCI"], default=["MA20", "MA50"])

    # Smart Interval Logic (Auto-fix invalid combinations)
    original_interval = interval
    if period in ["1mo", "3mo"] and interval in ["1m"]:
        interval = "1h" 
    elif period in ["6mo", "1y", "2y"] and interval in ["1m", "5m", "15m"]:
        interval = "1d" 
    elif period in ["5y", "10y", "max"] and interval in ["1m", "5m", "15m", "1h", "4h"]:
        interval = "1d" 
        
    if original_interval != interval:
        st.toast(f"⚠️ {period} periyodu için {original_interval} verisi mevcut değil. Otomatik olarak {interval} seçildi.", icon="ℹ️")

    # Futures Selector
    current_date = datetime.now()
    months_tr = {
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
        7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
    }
    
    # Calculate future contracts
    future_options = ["Spot (Hisse)"]
    future_map = {"Spot (Hisse)": symbol}
    
    for i in range(3):
        t_month = current_date.month + i
        t_year = current_date.year
        if t_month > 12:
            t_month -= 12
            t_year += 1
        
        # Construct a theoretical symbol or label
        # Note: YF symbols for VIOP are tricky. We'll use a placeholder logic.
        # If we knew the format: e.g. THYAOF2026.IS (Hypothetical)
        # For now, we will allow selection but might not find data.
        lbl = f"{months_tr[t_month]} {t_year} Vade"
        future_options.append(lbl)
        
        # Try to construct a symbol (Best Guess)
        # Format often: F_TICKERMMYY.IS or similar. 
        # Let's use a dummy format that likely won't fetch but shows intent.
        # If user provides correct format later, we can update.
        month_code = f"{t_month:02d}"
        year_short = str(t_year)[-2:]
        future_map[lbl] = f"F_{symbol.split('.')[0]}{month_code}{year_short}.IS"

    selected_asset_type = st.radio("Varlık Tipi", future_options, horizontal=True, label_visibility="collapsed")
    
    target_symbol = future_map[selected_asset_type]
    
    # If it's a future and not the spot, show a warning about data
    if selected_asset_type != "Spot (Hisse)":
        st.info(f"Seçilen Vade Sembolü: {target_symbol}. (Not: Yahoo Finance üzerinde VIOP verisi sınırlı olabilir.)")

    # Fetch
    df = get_stock_data(target_symbol, period, interval)
    
    if df is not None and not df.empty:
        # TradingView Style Chart
        # Determine subplot structure based on indicators
        has_rsi = "RSI" in indicators
        has_macd = "MACD" in indicators
        has_stoch = "Stoch" in indicators
        has_cci = "CCI" in indicators
        
        # Calculate how many subplots we need
        # Row 1: Price (always)
        # Row 2+: Indicators
        subplots = []
        if has_rsi: subplots.append("RSI")
        if has_macd: subplots.append("MACD")
        if has_stoch: subplots.append("Stoch")
        if has_cci: subplots.append("CCI")
        
        # If no subplots selected, show Volume
        if not subplots:
            subplots.append("Volume")
            
        row_heights = [0.6] + [0.4/len(subplots)] * len(subplots)
        
        fig = make_subplots(
            rows=1 + len(subplots), cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.02, 
            row_heights=row_heights
        )
        
        # Candlestick (Row 1)
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Fiyat",
            increasing_line_color='#81c995', increasing_fillcolor='#81c995',
            decreasing_line_color='#f28b82', decreasing_fillcolor='#f28b82'
        ), row=1, col=1)
        
        # Overlays (Row 1)
        if "MA20" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(20).mean(), name="MA20", line=dict(color='#2962ff', width=1)), row=1, col=1)
        if "MA50" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(), name="MA50", line=dict(color='#ff9800', width=1)), row=1, col=1)
        if "MA200" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), name="MA200", line=dict(color='#e91e63', width=1)), row=1, col=1)
        if "Bollinger" in indicators:
            ma = df['Close'].rolling(20).mean()
            std = df['Close'].rolling(20).std()
            fig.add_trace(go.Scatter(x=df.index, y=ma+2*std, name="BB Upper", line=dict(color='rgba(255,255,255,0.3)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=ma-2*std, name="BB Lower", line=dict(color='rgba(255,255,255,0.3)', width=1), fill='tonexty', fillcolor='rgba(255,255,255,0.05)'), row=1, col=1)
        if "Ichimoku" in indicators:
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
            high9 = df['High'].rolling(window=9).max()
            low9 = df['Low'].rolling(window=9).min()
            tenkan_sen = (high9 + low9) / 2

            # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
            high26 = df['High'].rolling(window=26).max()
            low26 = df['Low'].rolling(window=26).min()
            kijun_sen = (high26 + low26) / 2

            # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

            # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
            high52 = df['High'].rolling(window=52).max()
            low52 = df['Low'].rolling(window=52).min()
            senkou_span_b = ((high52 + low52) / 2).shift(26)

            # Chikou Span (Lagging Span): Close shifted back 26 periods
            chikou_span = df['Close'].shift(-26)

            fig.add_trace(go.Scatter(x=df.index, y=tenkan_sen, name="Tenkan", line=dict(color='#0496ff', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=kijun_sen, name="Kijun", line=dict(color='#991515', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=senkou_span_a, name="Span A", line=dict(color='rgba(0, 150, 0, 0.3)', width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=senkou_span_b, name="Span B", line=dict(color='rgba(150, 0, 0, 0.3)', width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', showlegend=False), row=1, col=1)

        # Latest Price Line & Label
        last_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else last_price
        price_color = '#81c995' if last_price >= prev_price else '#f28b82'
        
        fig.add_hline(
            y=last_price, 
            line_dash="dash", 
            line_color=price_color, 
            line_width=1,
            annotation_text=f"{last_price:.2f}",
            annotation_position="top right",
            annotation_font_size=11,
            annotation_font_color="white",
            annotation_bgcolor=price_color,
            row=1, col=1
        )


        # Subplots
        current_row = 2
        for ind in subplots:
            if ind == "RSI":
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color='#9c27b0')), row=current_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=current_row, col=1)
                
            elif ind == "MACD":
                exp1 = df['Close'].ewm(span=12).mean()
                exp2 = df['Close'].ewm(span=26).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9).mean()
                fig.add_trace(go.Scatter(x=df.index, y=macd, name="MACD", line=dict(color='#2962ff')), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=signal, name="Signal", line=dict(color='#ff9800')), row=current_row, col=1)
                fig.add_trace(go.Bar(x=df.index, y=macd-signal, name="Hist", marker_color='#787b86'), row=current_row, col=1)
                
            elif ind == "Stoch":
                # Stochastic Oscillator
                low14 = df['Low'].rolling(window=14).min()
                high14 = df['High'].rolling(window=14).max()
                k = 100 * ((df['Close'] - low14) / (high14 - low14))
                d = k.rolling(window=3).mean()
                fig.add_trace(go.Scatter(x=df.index, y=k, name="%K", line=dict(color='#2962ff')), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=d, name="%D", line=dict(color='#ff9800')), row=current_row, col=1)
                fig.add_hline(y=80, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=current_row, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=current_row, col=1)
                
            elif ind == "CCI":
                # Commodity Channel Index
                tp = (df['High'] + df['Low'] + df['Close']) / 3
                sma = tp.rolling(20).mean()
                mad = tp.rolling(20).apply(lambda x: pd.Series(x).sub(x.mean()).abs().mean())
                cci = (tp - sma) / (0.015 * mad)
                fig.add_trace(go.Scatter(x=df.index, y=cci, name="CCI", line=dict(color='#00bcd4')), row=current_row, col=1)
                fig.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=current_row, col=1)
                fig.add_hline(y=-100, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=current_row, col=1)
                
            elif ind == "Volume":
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Hacim", marker_color='rgba(41, 98, 255, 0.3)'), row=current_row, col=1)
            
            current_row += 1

        # Layout Config
        fig.update_layout(
            height=800 if len(subplots) > 1 else 600,
            template="plotly_dark",
            paper_bgcolor='#131314',
            plot_bgcolor='#131314',
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=50, t=20, b=20),
            xaxis=dict(showgrid=True, gridcolor='#2c2d2f'),
            yaxis=dict(showgrid=True, gridcolor='#2c2d2f', side='right'),
            legend=dict(orientation="h", y=1, x=0, bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats Grid
        st.markdown("#### İstatistikler")
        last = df.iloc[-1]
        
        # Helper to safely get float value
        def get_val(series_val):
            if isinstance(series_val, pd.Series):
                return float(series_val.iloc[0])
            return float(series_val)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Kapanış", f"{get_val(last['Close']):.2f}")
        s2.metric("Açılış", f"{get_val(last['Open']):.2f}")
        s3.metric("Yüksek", f"{get_val(last['High']):.2f}")
        s4.metric("Düşük", f"{get_val(last['Low']):.2f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # CSV Download
        csv = df.to_csv().encode('utf-8')
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="Verileri indir (CSV)",
                data=csv,
                file_name=f"{symbol}_data.csv",
                mime="text/csv",
                key='download-csv'
            )
            
        with col_dl2:
            analysis_df = calculate_technical_signals(df)
            if not analysis_df.empty:
                analysis_csv = analysis_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Analizi indir (CSV)",
                    data=analysis_csv,
                    file_name=f"{symbol}_analiz.csv",
                    mime="text/csv",
                    key='download-analysis-csv'
                )
                
        # Display Analysis Table
        if not analysis_df.empty:
            st.markdown("#### Teknik analiz özeti")
            
            def color_signal(val):
                color = '#e3e3e3'
                if 'GÜÇLÜ AL' in str(val) or ('AL' in str(val) and 'NORMAL' not in str(val)):
                    color = '#81c995'
                elif 'GÜÇLÜ SAT' in str(val) or 'SAT' in str(val):
                    color = '#f28b82'
                elif 'BEKLE' in str(val):
                    color = '#8ab4f8'
                return f'color: {color}; font-weight: bold'

            st.dataframe(
                analysis_df.style.map(color_signal, subset=['Sinyal']),
                use_container_width=True,
                hide_index=True
            )

        # Futures (VIOP) Section - REMOVED (Moved to top selector)
        # st.markdown("<br>", unsafe_allow_html=True)
        # st.markdown("### 🗓️ Vadeli İşlemler (VIOP)")
        # ... (Removed previous list implementation)
        
        # Calculate next 3 months
        current_date = datetime.now()
        futures_data = []
        
        months_tr = {
            1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
            7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
        }
        
        for i in range(3):
            # Calculate target month
            target_month = current_date.month + i
            target_year = current_date.year
            
            if target_month > 12:
                target_month -= 12
                target_year += 1
                
            month_name = months_tr[target_month]
            contract_name = f"{symbol.split('.')[0]} - {month_name} {target_year} Vadesi"
            
            # Placeholder for data (since YF doesn't reliably support BIST single stock futures)
            futures_data.append({
                "Vade": f"{month_name} {target_year}",
                "Sözleşme": contract_name,
                "Durum": "Aktif",
                "Fiyat": "N/A" # Placeholder
            })
            
        
        st.markdown('<hr class="gf-divider">', unsafe_allow_html=True)
        st.markdown("#### Vadeli sözleşmeler (VİOP)")
        st.dataframe(
            pd.DataFrame(futures_data),
            use_container_width=True,
            hide_index=True
        )

def render_bulk_analysis():
    render_topbar(show_search=False)
    b1, b2 = st.columns([0.7, 9])
    with b1:
        if st.button("← Geri", key="bulk_back"):
            go_to_home()
    with b2:
        st.markdown("### Toplu teknik analiz")
        
    st.markdown("Analiz etmek istediğiniz hisseleri seçin ve raporu oluşturun.")
    
    c_mode, c_dummy = st.columns([1, 2])
    with c_mode:
        analysis_mode = st.radio("Rapor Tipi", ["Basit", "Detaylı"], horizontal=True)
    
    selected_symbols = st.multiselect(
        "Hisseler", 
        BIST100_SYMBOLS, 
        default=BIST100_SYMBOLS[:5] # Default first 5
    )
    
    if st.button("Analiz et", type="primary"):
        if not selected_symbols:
            st.warning("Lütfen en az bir hisse seçin.")
            return
            
        progress_bar = st.progress(0)
        results = []
        
        for idx, symbol in enumerate(selected_symbols):
            # Fetch data (1 year daily)
            df = get_stock_data(symbol, "1y", "1d")
            if df is not None and not df.empty:
                # Calculate signals
                signals_df = calculate_technical_signals(df)
                
                # Extract key signals for summary
                summary = {"Sembol": symbol, "Son Fiyat": df['Close'].iloc[-1]}
                
                # Flatten signals
                for _, row in signals_df.iterrows():
                    ind_name = row['İndikatör'].split(' ')[0] # Short name
                    summary[ind_name] = row['Sinyal']
                    
                    if analysis_mode == "Detaylı":
                        # Extract value from "Değerler" string or use RawData if we added it
                        # For now, let's just add the full description string as a column
                        summary[f"{ind_name} Detay"] = row['Değerler']
                    
                results.append(summary)
            
            progress_bar.progress((idx + 1) / len(selected_symbols))
            
        if results:
            res_df = pd.DataFrame(results)
            
            st.success("Analiz Tamamlandı!")
            
            # Display Table
            def color_bulk(val):
                color = ''
                if 'AL' in str(val): color = 'color: #81c995; font-weight: 600'
                elif 'SAT' in str(val): color = 'color: #f28b82; font-weight: 600'
                elif 'BEKLE' in str(val): color = 'color: #8ab4f8'
                return color

            st.dataframe(
                res_df.style.map(color_bulk),
                use_container_width=True,
                hide_index=True
            )
            
            # CSV Download
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Raporu indir (CSV)",
                data=csv,
                file_name=f"toplu_analiz_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.error("Veri alınamadı.")

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


# --- DAILY LOG / EXCEL TABLOSU ---

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


def maybe_run_auto_capture():
    """
    Türkiye saati 18:30'dan sonraysa ve bugün için kayıt yoksa otomatik olarak
    günlük kayıt yapar. Bu fonksiyon her sayfa render'ında çağrılır; aynı session
    içinde tekrar çalışmasın diye st.session_state.last_auto_check ile sınırlanır.
    """
    try:
        now = dl.now_istanbul()
        today_str = now.date().isoformat()

        last_check = st.session_state.get('last_auto_check_date')
        if last_check == today_str:
            return None

        # 18:30 ve sonrası
        if (now.hour, now.minute) < (18, 30):
            return None

        # Bugün için zaten kayıt varsa çalıştırma
        if dl.has_record_for_today():
            st.session_state.last_auto_check_date = today_str
            return None

        symbols = dl.get_target_symbols()
        if not symbols:
            return None

        result = run_daily_capture(symbols, overwrite=False)
        st.session_state.last_auto_check_date = today_str
        st.session_state.last_auto_capture_result = result
        return result
    except Exception:
        return None


def render_daily_log():
    render_topbar(show_search=False)
    c_back, c_refresh = st.columns([1, 1])
    with c_back:
        if st.button("← Geri", key="daily_log_back"):
            go_to_home()
    st.markdown("### Günlük kayıt / Excel tablosu")
    with c_refresh:
        # Sayfa açıkken her 5 dakikada bir tetikle (otomatik 18:30 kontrolü için)
        st_autorefresh(interval=300_000, limit=None, key="daily_log_autorefresh")

    # Otomatik 18:30 kontrolü
    auto_result = maybe_run_auto_capture()
    if auto_result and auto_result.get('inserted'):
        st.success(
            f"⏰ Otomatik kayıt yapıldı (18:30 sonrası): "
            f"{auto_result['inserted']} eklendi, {auto_result['skipped']} atlandı, "
            f"{len(auto_result['failed'])} başarısız."
        )

    # Bilgi paneli
    symbols = dl.get_target_symbols()
    now_ist = dl.now_istanbul()
    today_str = now_ist.date().isoformat()

    info_cols = st.columns(4)
    info_cols[0].metric("Bugün (TR)", today_str)
    info_cols[1].metric("Saat (TR)", now_ist.strftime("%H:%M"))
    info_cols[2].metric("Takipteki hisseler", str(len(symbols)))
    info_cols[3].metric("Toplam kayıt", str(dl.total_record_count()))

    with st.expander("Takip edilen hisse listesi (Excel + ek hisseler)"):
        st.write(", ".join(symbols) if symbols else "Hisse bulunamadı.")
        st.caption(
            "Excel dosyası: `Başlıksız e-tablo.xlsx` (proje kökünde). "
            "Sütun başlıklarındaki 3-6 karakterli hisse kodları otomatik algılanır. "
            "Her durumda SISE ve ASELS dahil edilir."
        )

    st.markdown("---")
    st.markdown("#### Manuel kayıt")

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        manual_btn = st.button("Bugünün verilerini kaydet", type="primary", use_container_width=True)
    with c2:
        overwrite_today = st.checkbox(
            "Bugünkü mevcut kayıtları güncelle",
            value=False,
            help="İşaretliyse bugün için aynı hisseye ait kayıt varsa güncellenir. Geçmiş tarihler ASLA değişmez."
        )
    with c3:
        st.caption("Otomatik kayıt: Türkiye saati ile **18:30** sonrasında, sayfa açıkken bir kez çalışır.")

    if manual_btn:
        if not symbols:
            st.error("Takip edilecek hisse bulunamadı. Excel dosyasını kontrol edin.")
        else:
            progress_bar = st.progress(0.0, text="Veriler hazırlanıyor...")
            status = st.empty()

            def _cb(p, s):
                progress_bar.progress(min(1.0, p), text=f"İşleniyor: {s}")
                status.caption(f"İşlenen: {s}")

            result = run_daily_capture(symbols, overwrite=overwrite_today, progress_cb=_cb)
            progress_bar.empty()
            status.empty()

            msg = (
                f"✅ Manuel kayıt tamamlandı: **{result['inserted']}** kayıt eklendi/güncellendi, "
                f"**{result['skipped']}** zaten mevcuttu (atlandı), "
                f"**{len(result['failed'])}** hisse veri çekilemediği için başarısız oldu."
            )
            st.success(msg)
            if result['failed']:
                st.warning("Başarısız: " + ", ".join(result['failed']))
            st.rerun()

    st.markdown("---")
    st.markdown("#### Tablo")

    df = dl.fetch_all()

    if df.empty:
        st.info("Henüz kayıt bulunmuyor. Üstteki **Bugünün Verilerini Kaydet** butonuyla başlayabilirsiniz.")
        return

    # Filtreler
    fc1, fc2, fc3 = st.columns([2, 2, 2])
    with fc1:
        date_options = ["(Tümü)"] + sorted(df['Tarih'].dropna().unique().tolist(), reverse=True)
        sel_date = st.selectbox("Tarihe göre filtrele", date_options, index=0)
    with fc2:
        sym_options = ["(Tümü)"] + sorted(df['Hisse'].dropna().unique().tolist())
        sel_sym = st.selectbox("Hisseye göre filtrele", sym_options, index=0)
    with fc3:
        text_filter = st.text_input("Metin ara (sinyallerde)", "")

    view_df = df.copy()
    if sel_date != "(Tümü)":
        view_df = view_df[view_df['Tarih'] == sel_date]
    if sel_sym != "(Tümü)":
        view_df = view_df[view_df['Hisse'] == sel_sym]
    if text_filter:
        mask = view_df.apply(
            lambda row: row.astype(str).str.contains(text_filter, case=False, na=False).any(),
            axis=1,
        )
        view_df = view_df[mask]

    st.caption(f"Görünen kayıt: {len(view_df)} / Toplam: {len(df)}")

    # Sinyalleri renklendirme
    def _color_signal(val):
        s = str(val).upper()
        if 'GÜÇLÜ AL' in s:
            return 'color: #81c995; font-weight: 700'
        if 'GÜÇLÜ SAT' in s:
            return 'color: #f28b82; font-weight: 700'
        if 'AL' in s and 'SAT' not in s:
            return 'color: #81c995'
        if 'SAT' in s:
            return 'color: #f28b82'
        if 'BEKLE' in s:
            return 'color: #8ab4f8'
        return ''

    signal_cols = ['MA Sinyali', 'MACD Sinyali', 'Bollinger Sinyali', 'Stokastik', 'Ichimoku', 'Genel Sinyal']
    signal_cols = [c for c in signal_cols if c in view_df.columns]

    styler = view_df.style.format({
        'Kapanış': '{:.2f}',
        'Açılış': '{:.2f}',
        'Yüksek': '{:.2f}',
        'Düşük': '{:.2f}',
        'RSI': '{:.2f}',
    }, na_rep='-').map(_color_signal, subset=signal_cols)

    st.dataframe(styler, use_container_width=True, hide_index=True, height=520)

    # İndirme butonları
    st.markdown("#### İndir")
    dc1, dc2 = st.columns(2)
    with dc1:
        csv_bytes = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Tüm kayıtları indir (CSV)",
            data=csv_bytes,
            file_name=f"borsabot_gunluk_kayit_{today_str}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dc2:
        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Günlük Kayıt', index=False)
        st.download_button(
            label="Tüm kayıtları indir (XLSX)",
            data=xlsx_buf.getvalue(),
            file_name=f"borsabot_gunluk_kayit_{today_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


# --- MAIN ---
# DB'yi başlat (uygulama ilk açıldığında tabloyu oluşturur)
try:
    dl.init_db()
except Exception:
    pass

# 18:30 sonrasında otomatik günlük kayıt; her sayfa render'ında bir kez
# kontrol edilir, aynı gün içinde tekrar tetiklenmez.
maybe_run_auto_capture()

if st.session_state.view == 'home':
    render_home()
elif st.session_state.view == 'detail':
    render_detail()
elif st.session_state.view == 'bulk':
    render_bulk_analysis()
elif st.session_state.view == 'daily_log':
    render_daily_log()
elif st.session_state.view == 'forecast':
    render_forecast()
