import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import requests
from bs4 import BeautifulSoup
import ssl

# SSL Fix
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

# --- CONFIGURATION ---
st.set_page_config(
    page_title="BorsaBot Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- TRADINGVIEW THEME CSS ---
st.markdown("""
<style>
    /* Global Reset & Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background - TradingView Black */
    .stApp {
        background-color: #131722;
        color: #d1d4dc;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e222d;
        border-right: 1px solid #2a2e39;
    }
    
    /* Top Market Bar */
    .market-bar {
        display: flex;
        align-items: center;
        background-color: #1e222d;
        padding: 10px 20px;
        border-bottom: 1px solid #2a2e39;
        margin-top: -50px; /* Pull up to hide default header space if possible */
        margin-left: -5rem;
        margin-right: -5rem;
        padding-left: 5rem;
        overflow-x: auto;
        white-space: nowrap;
        gap: 30px;
    }
    .market-item {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    .market-symbol {
        font-size: 0.75rem;
        color: #787b86;
        font-weight: 600;
    }
    .market-price-group {
        display: flex;
        align-items: baseline;
        gap: 8px;
    }
    .market-price {
        font-size: 0.95rem;
        color: #d1d4dc;
        font-weight: 600;
    }
    .market-change-pos { color: #26a69a; font-size: 0.85rem; }
    .market-change-neg { color: #ef5350; font-size: 0.85rem; }
    
    /* Cards & Containers */
    .tv-card {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 4px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #d1d4dc !important;
        font-weight: 600;
    }
    
    /* Inputs */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #2a2e39 !important;
        color: #d1d4dc !important;
        border: 1px solid #363a45 !important;
        border-radius: 4px;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #2962ff;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s;
    }
    .stButton button:hover {
        background-color: #1e53e5;
    }
    
    /* News Feed */
    .news-item {
        padding: 12px 0;
        border-bottom: 1px solid #2a2e39;
        display: flex;
        gap: 15px;
        align-items: start;
    }
    .news-item:last-child { border-bottom: none; }
    .news-img-small {
        width: 80px;
        height: 60px;
        object-fit: cover;
        border-radius: 4px;
    }
    .news-content { flex: 1; }
    .news-title {
        font-size: 0.95rem;
        color: #d1d4dc;
        text-decoration: none;
        font-weight: 500;
        display: block;
        margin-bottom: 4px;
    }
    .news-title:hover { color: #2962ff; }
    .news-time {
        font-size: 0.75rem;
        color: #787b86;
    }
    
    /* Table Overrides */
    [data-testid="stDataFrame"] {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
    }
    [data-testid="stDataFrame"] th {
        background-color: #2a2e39 !important;
        color: #d1d4dc !important;
    }
    [data-testid="stDataFrame"] td {
        color: #d1d4dc !important;
        border-bottom: 1px solid #2a2e39 !important;
    }
    
    /* Last Viewed Chips */
    .chip-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 20px;
    }
    .chip {
        background-color: #2a2e39;
        color: #d1d4dc;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.85rem;
        cursor: pointer;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    .chip:hover {
        border-color: #2962ff;
        color: #2962ff;
    }
</style>
""", unsafe_allow_html=True)

# --- FULL STOCK LIST ---
BIST100_SYMBOLS = [
    "XU030.IS", "XU100.IS",  # Endeksler
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS",
    "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KOZAL.IS", "KOZAA.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
    "SASA.IS", "SISE.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", "TSKB.IS", "TUPRS.IS",
    "VAKBN.IS", "YKBNK.IS", "YUNSA.IS", "ZOREN.IS", "AHLAT.IS", "AKSEN.IS", "ALBRK.IS", "ALCAR.IS",
    "ALGYO.IS", "ALKIM.IS", "ANACM.IS", "ASUZU.IS", "AYCES.IS", "BAGFS.IS", "BASCM.IS", "BERA.IS",
    "BIENY.IS", "BRISA.IS", "BRYAT.IS", "BUCIM.IS", "CCOLA.IS", "CEMAS.IS", "CEMTS.IS", "CIMSA.IS",
    "CUSAN.IS", "DOHOL.IS", "EGEEN.IS", "ENJSA.IS", "ENKAI.IS", "FMIZP.IS", "GESAN.IS", "GLYHO.IS",
    "GSDHO.IS", "GSDDE.IS", "HALKB.IS", "HATEK.IS", "IPEKE.IS", "ISDMR.IS", "KAREL.IS", "KARSN.IS",
    "KONTR.IS", "KONYA.IS", "KORDS.IS", "KOZAA.IS", "KRDMD.IS", "LOGO.IS", "MGROS.IS", "NETAS.IS",
    "NTHOL.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "POLHO.IS", "PRKAB.IS",
    "PRKME.IS", "QUAGR.IS", "SAFKN.IS", "SELEC.IS", "SELGD.IS", "SISE.IS", "SKBNK.IS", "SMRTG.IS",
    "SNGYO.IS", "SOKM.IS", "TATGD.IS", "TCELL.IS", "TKURU.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS",
    "TTRAK.IS", "TUPRS.IS", "ULKER.IS", "VAKBN.IS", "VESTL.IS", "YATAS.IS", "YKBNK.IS", "YUNSA.IS"
]

# --- STATE MANAGEMENT ---
if 'view' not in st.session_state:
    st.session_state.view = 'home'
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'THYAO.IS'
if 'last_viewed' not in st.session_state:
    st.session_state.last_viewed = []

def go_to_detail(symbol):
    st.session_state.selected_symbol = symbol
    if symbol in st.session_state.last_viewed:
        st.session_state.last_viewed.remove(symbol)
    st.session_state.last_viewed.insert(0, symbol)
    if len(st.session_state.last_viewed) > 12:
        st.session_state.last_viewed.pop()
    st.session_state.view = 'detail'
    st.rerun()

def go_to_home():
    st.session_state.view = 'home'
    st.rerun()

def go_to_bulk():
    st.session_state.view = 'bulk'
    st.rerun()

# --- DATA FETCHING ---
@st.cache_data(ttl=60)
def get_market_summary():
    tickers = ["USDTRY=X", "EURTRY=X", "EURUSD=X", "GC=F", "SI=F", "XU100.IS"]
    try:
        df = yf.download(tickers, period="2d", group_by='ticker', progress=False)
        summary = {}
        names = {
            "USDTRY=X": "Dolar", 
            "EURTRY=X": "Euro", 
            "EURUSD=X": "Parite", 
            "GC=F": "Altın", 
            "SI=F": "Gümüş", 
            "XU100.IS": "BIST 100"
        }
        
        for ticker in tickers:
            try:
                hist = df[ticker]
                if len(hist) >= 1:
                    close = hist['Close'].iloc[-1]
                    # Check for NaN
                    if pd.isna(close):
                        continue
                        
                    change = 0.0
                    if len(hist) >= 2:
                        prev = hist['Close'].iloc[-2]
                        if not pd.isna(prev) and prev != 0:
                            change = ((close - prev) / prev * 100)
                            
                    summary[names[ticker]] = {"price": close, "change": change}
            except:
                continue
        return summary
    except:
        return {}

@st.cache_data(ttl=300)
def get_market_movers():
    tickers = [s for s in BIST100_SYMBOLS if "XU" not in s][:30]
    data = []
    try:
        df = yf.download(tickers, period="2d", group_by='ticker', progress=False)
        for ticker in tickers:
            try:
                hist = df[ticker]
                if len(hist) >= 2:
                    close = hist['Close'].iloc[-1]
                    change = ((close - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                    data.append({"Symbol": ticker, "Price": close, "Change": change})
            except:
                continue
        return pd.DataFrame(data).sort_values("Change", ascending=False)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
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
            image_url = "https://via.placeholder.com/80x60?text=News"
            image_tag = item.find('image')
            if image_tag: image_url = image_tag.text
            
            news_list.append({'title': title, 'link': link, 'published': pub_date, 'image': image_url})
        return news_list
    except:
        return []

def get_stock_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        # Flatten MultiIndex if present (yfinance update)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return None

# --- COMPONENTS ---

def render_market_bar():
    market_data = get_market_summary()
    if market_data:
        html = '<div class="market-bar">'
        for key, data in market_data.items():
            color_class = "market-change-pos" if data['change'] >= 0 else "market-change-neg"
            sign = "+" if data['change'] >= 0 else ""
            price_display = f"{data['price']:.2f}" if not pd.isna(data['price']) else "N/A"
            change_display = f"{sign}{data['change']:.2f}%" if not pd.isna(data['change']) else "0.00%"
            
            html += f"""
<div class="market-item">
    <span class="market-symbol">{key}</span>
    <div class="market-price-group">
        <span class="market-price">{price_display}</span>
        <span class="{color_class}">{change_display}</span>
    </div>
</div>
"""
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

def render_home():
    render_market_bar()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Layout: 2 Columns (Left: Dashboard, Right: News Feed)
    col_main, col_side = st.columns([3, 1])
    
    with col_main:
        # Search & History
        c_search, c_bulk = st.columns([3, 1])
        with c_search:
            st.markdown("### 🔎 Piyasa Takibi")
        with c_bulk:
            if st.button("📊 Toplu Analiz", use_container_width=True):
                go_to_bulk()
        
        # Search
        selected = st.selectbox("Sembol Ara", BIST100_SYMBOLS, index=None, placeholder="Hisse senedi arayın...", label_visibility="collapsed")
        if selected:
            go_to_detail(selected)
            
        # Last Viewed Chips
        if st.session_state.last_viewed:
            st.markdown('<div class="chip-container">', unsafe_allow_html=True)
            cols = st.columns(len(st.session_state.last_viewed))
            for idx, symbol in enumerate(st.session_state.last_viewed):
                if idx < 8:
                    if cols[idx].button(symbol, key=f"chip_{symbol}"):
                        go_to_detail(symbol)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Movers Tables
        st.markdown("<br>", unsafe_allow_html=True)
        movers = get_market_movers()
        if not movers.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🚀 Yükselenler")
                st.dataframe(
                    movers.head(10)[['Symbol', 'Price', 'Change']].style.format({'Price': '{:.2f}', 'Change': '{:+.2f}%'})
                    .applymap(lambda x: 'color: #26a69a', subset=['Change']),
                    use_container_width=True,
                    hide_index=True
                )
            with c2:
                st.markdown("#### 🔻 Düşenler")
                st.dataframe(
                    movers.tail(10).sort_values("Change", ascending=True)[['Symbol', 'Price', 'Change']].style.format({'Price': '{:.2f}', 'Change': '{:+.2f}%'})
                    .applymap(lambda x: 'color: #ef5350', subset=['Change']),
                    use_container_width=True,
                    hide_index=True
                )

    with col_side:
        st.markdown("### 📰 Haber Akışı")
        news = get_bloomberg_news()
        if news:
            for item in news[:10]:
                st.markdown(f"""
                <div class="news-item">
                    <div class="news-content">
                        <a href="{item['link']}" target="_blank" class="news-title">{item['title']}</a>
                        <div class="news-time">{item['published'][5:16]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def render_detail():
    symbol = st.session_state.selected_symbol
    
    # Top Control Bar
    c1, c2 = st.columns([1, 10])
    with c1:
        if st.button("⬅"):
            go_to_home()
    with c2:
        st.markdown(f"## {symbol}")

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
            increasing_line_color='#26a69a', increasing_fillcolor='#26a69a',
            decreasing_line_color='#ef5350', decreasing_fillcolor='#ef5350'
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
        price_color = '#26a69a' if last_price >= prev_price else '#ef5350'
        
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
            paper_bgcolor='#131722',
            plot_bgcolor='#131722',
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=50, t=20, b=20),
            xaxis=dict(showgrid=True, gridcolor='#2a2e39'),
            yaxis=dict(showgrid=True, gridcolor='#2a2e39', side='right'),
            legend=dict(orientation="h", y=1, x=0, bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats Grid
        st.markdown("### İstatistikler")
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
                label="📥 Verileri İndir (CSV)",
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
                    label="📥 Analiz İndir (CSV)",
                    data=analysis_csv,
                    file_name=f"{symbol}_analiz.csv",
                    mime="text/csv",
                    key='download-analysis-csv'
                )
                
        # Display Analysis Table
        if not analysis_df.empty:
            st.markdown("### 📈 Teknik Analiz Özeti")
            
            def color_signal(val):
                color = '#d1d4dc'
                if 'GÜÇLÜ AL' in str(val) or ('AL' in str(val) and 'NORMAL' not in str(val)):
                    color = '#26a69a'
                elif 'GÜÇLÜ SAT' in str(val) or 'SAT' in str(val):
                    color = '#ef5350'
                elif 'BEKLE' in str(val):
                    color = '#2962ff'
                return f'color: {color}; font-weight: bold'

            st.dataframe(
                analysis_df.style.applymap(color_signal, subset=['Sinyal']),
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
            
        
        st.dataframe(
            pd.DataFrame(futures_data),
            use_container_width=True,
            hide_index=True
        )

def render_bulk_analysis():
    st.markdown("## 📊 Toplu Teknik Analiz")
    if st.button("⬅ Ana Sayfaya Dön"):
        go_to_home()
        
    st.markdown("Analiz etmek istediğiniz hisseleri seçin ve raporu oluşturun.")
    
    c_mode, c_dummy = st.columns([1, 2])
    with c_mode:
        analysis_mode = st.radio("Rapor Tipi", ["Basit", "Detaylı"], horizontal=True)
    
    selected_symbols = st.multiselect(
        "Hisseler", 
        BIST100_SYMBOLS, 
        default=BIST100_SYMBOLS[:5] # Default first 5
    )
    
    if st.button("🚀 Analiz Et", type="primary"):
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
                if 'AL' in str(val): color = 'color: #26a69a; font-weight: bold'
                elif 'SAT' in str(val): color = 'color: #ef5350; font-weight: bold'
                elif 'BEKLE' in str(val): color = 'color: #2962ff'
                return color

            st.dataframe(
                res_df.style.applymap(color_bulk),
                use_container_width=True,
                hide_index=True
            )
            
            # CSV Download
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Raporu İndir (CSV)",
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

# --- MAIN ---
if st.session_state.view == 'home':
    render_home()
elif st.session_state.view == 'detail':
    render_detail()
elif st.session_state.view == 'bulk':
    render_bulk_analysis()
