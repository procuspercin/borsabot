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
        st.markdown("### 🔎 Piyasa Takibi")
        
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
                    <img src="{item['image']}" class="news-img-small">
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

    # Fetch
    df = get_stock_data(symbol, period, interval)
    
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
        st.download_button(
            label="📥 Verileri İndir (CSV)",
            data=csv,
            file_name=f"{symbol}_data.csv",
            mime="text/csv",
            key='download-csv'
        )

# --- MAIN ---
if st.session_state.view == 'home':
    render_home()
elif st.session_state.view == 'detail':
    render_detail()
