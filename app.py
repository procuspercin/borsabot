import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
from streamlit_autorefresh import st_autorefresh

# Sayfa yapılandırması
st.set_page_config(
    page_title="Borsa Teknik Analiz",
    page_icon="📈",
    layout="wide"
)

# Başlık
st.title("Borsa Teknik Analiz Uygulaması")

# Sidebar - Sembol ve Zaman Aralığı Seçimi
st.sidebar.header("Ayarlar")

# Otomatik yenileme (her 60 saniyede bir)
st_autorefresh(interval=60 * 1000, key="datarefresh")

# BIST 100 hisseleri
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

# Sembol seçimi
selected_symbol = st.sidebar.selectbox("Sembol Seçin", BIST100_SYMBOLS)

# Zaman aralığı seçimi
timeframe = st.sidebar.selectbox("Zaman Dilimi", ["1d", "4h", "1h", "15m"], index=3)

# Periyot seçimi
period = st.sidebar.selectbox("Periyot", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])

# Sidebar - Tarih seçici ekle
st.sidebar.header("Tarih Seçimi")
today = datetime.today().date()
min_date = today - timedelta(days=365*5)
selected_date = st.sidebar.date_input("Tarih Seçin", value=today, min_value=min_date, max_value=today)

# Fiyat grafiği gösterme seçeneği
show_price_chart = st.sidebar.checkbox("Fiyat Grafiğini Göster", value=True)

# İndikatör seçimi
st.sidebar.header("İndikatörler")
indicators = {
    "MA": "Hareketli Ortalama",
    "MACD": "MACD",
    "BB": "Bollinger Bantları",
    "RSI": "RSI",
    "STOCH": "Stokastik Osilatör",
    "FIB": "Fibonacci Düzeltmesi",
    "ICHIMOKU": "Ichimoku Bulutu",
    "STD": "Standard Sapma"
}

selected_indicators = []
for key, name in indicators.items():
    if st.sidebar.checkbox(name, key=key):
        selected_indicators.append(key)

# Sidebar - Günlük fiyat tablosu gösterme seçeneği
show_price_table = st.sidebar.checkbox("Günlük Fiyat Tablosunu Göster", value=True)

# Veri çekme fonksiyonu
def get_data(symbol, period, interval):
    try:
        # Bugünün tarihini end olarak kullan
        today_str = datetime.today().strftime('%Y-%m-%d')
        data = yf.download(symbol, period=period, interval=interval, end=today_str)
        if data.empty:
            st.error(f"{symbol} için veri bulunamadı!")
            return None
        return data
    except Exception as e:
        st.error(f"Veri çekilirken hata oluştu: {str(e)}")
        return None

# İndikatör hesaplama fonksiyonu
def calculate_indicators(df, selected_indicators):
    # Veri çerçevesini kopyala
    df = df.copy()
    
    for indicator in selected_indicators:
        if indicator == "MA":
            # Hareketli ortalamaları hesapla
            df['ma20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['ma50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['ma200'] = df['Close'].rolling(window=200, min_periods=1).mean()
            
        elif indicator == "MACD":
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']
            
        elif indicator == "BB":
            # Bollinger Bands hesapla
            window = 20
            df['bb_mid'] = df['Close'].rolling(window=window, min_periods=1).mean()
            df['bb_std'] = df['Close'].rolling(window=window, min_periods=1).std()
            
            # Üst ve alt bantları hesapla
            df['bb_high'] = df['bb_mid'] + (df['bb_std'] * 2)
            df['bb_low'] = df['bb_mid'] - (df['bb_std'] * 2)
            
            # NaN değerleri temizle
            df['bb_mid'] = df['bb_mid'].fillna(method='ffill')
            df['bb_high'] = df['bb_high'].fillna(method='ffill')
            df['bb_low'] = df['bb_low'].fillna(method='ffill')
            
            # Geçici sütunu sil
            df.drop('bb_std', axis=1, inplace=True)
            
        elif indicator == "RSI":
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
        elif indicator == "STOCH":
            low_min = df['Low'].rolling(window=14).min()
            high_max = df['High'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
        elif indicator == "FIB":
            high = df['High'].rolling(window=20).max()
            low = df['Low'].rolling(window=20).min()
            diff = high - low
            df['fib_0'] = low
            df['fib_0.236'] = low + 0.236 * diff
            df['fib_0.382'] = low + 0.382 * diff
            df['fib_0.5'] = low + 0.5 * diff
            df['fib_0.618'] = low + 0.618 * diff
            df['fib_0.786'] = low + 0.786 * diff
            df['fib_1'] = high
            
        elif indicator == "ICHIMOKU":
            # Tenkan-sen (Conversion Line)
            high_9 = df['High'].rolling(window=9).max()
            low_9 = df['Low'].rolling(window=9).min()
            df['ichi_a'] = (high_9 + low_9) / 2
            
            # Kijun-sen (Base Line)
            high_26 = df['High'].rolling(window=26).max()
            low_26 = df['Low'].rolling(window=26).min()
            df['ichi_b'] = (high_26 + low_26) / 2
            
            # Senkou Span A (Leading Span A)
            df['ichi_base'] = ((df['ichi_a'] + df['ichi_b']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            high_52 = df['High'].rolling(window=52).max()
            low_52 = df['Low'].rolling(window=52).min()
            df['ichi_conv'] = ((high_52 + low_52) / 2).shift(26)
            
        elif indicator == "STD":
            df['std'] = df['Close'].rolling(window=20).std()
    
    return df

# Grafik çizme fonksiyonu
def plot_chart(df, symbol, selected_indicators, show_price_chart=True):
    if df is None or df.empty:
        st.error("Grafik çizilemiyor: Veri bulunamadı!")
        return None
        
    num_plots = 1  # Fiyat grafiği her zaman var
    if selected_indicators:
        num_plots += len(selected_indicators)
        
    fig = make_subplots(
        rows=num_plots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Fiyat"] + [indicators[ind] for ind in selected_indicators] if selected_indicators else ["Fiyat"]
    )
    
    plot_row = 1
    
    # Fiyat grafiği her zaman ilk plot olarak eklenecek
    if show_price_chart:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Fiyat"
            ),
            row=plot_row, col=1
        )
        
        # Fiyat grafiğine hacim ekle
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Hacim",
                marker_color='rgba(0,0,255,0.3)'
            ),
            row=plot_row, col=1
        )
    
    plot_row += 1
    
    # İndikatörleri ekle
    for indicator in selected_indicators:
        if indicator == "MA":
            fig.add_trace(go.Scatter(x=df.index, y=df['ma20'], name="MA20", line=dict(color='blue')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ma50'], name="MA50", line=dict(color='red')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ma200'], name="MA200", line=dict(color='green')), row=plot_row, col=1)
        elif indicator == "MACD":
            fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name="MACD", line=dict(color='blue')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name="Signal", line=dict(color='red')), row=plot_row, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['macd_diff'], name="Histogram", marker_color='gray'), row=plot_row, col=1)
        elif indicator == "BB":
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_high'], name="Üst Bant", line=dict(color='red', dash='dash')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_mid'], name="Orta Bant", line=dict(color='blue', dash='dash')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_low'], name="Alt Bant", line=dict(color='green', dash='dash')), row=plot_row, col=1)
        elif indicator == "RSI":
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name="RSI", line=dict(color='blue')), row=plot_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=plot_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=plot_row, col=1)
        elif indicator == "STOCH":
            fig.add_trace(go.Scatter(x=df.index, y=df['stoch_k'], name="%K", line=dict(color='blue')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['stoch_d'], name="%D", line=dict(color='red')), row=plot_row, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=plot_row, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=plot_row, col=1)
        elif indicator == "FIB":
            fig.add_trace(go.Scatter(x=df.index, y=df['fib_0'], name="0%", line=dict(color='black')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['fib_0.236'], name="23.6%", line=dict(color='blue')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['fib_0.382'], name="38.2%", line=dict(color='green')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['fib_0.5'], name="50%", line=dict(color='yellow')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['fib_0.618'], name="61.8%", line=dict(color='orange')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['fib_0.786'], name="78.6%", line=dict(color='red')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['fib_1'], name="100%", line=dict(color='purple')), row=plot_row, col=1)
        elif indicator == "ICHIMOKU":
            fig.add_trace(go.Scatter(x=df.index, y=df['ichi_a'], name="Tenkan-sen", line=dict(color='blue')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ichi_b'], name="Kijun-sen", line=dict(color='red')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ichi_base'], name="Senkou Span A", line=dict(color='green')), row=plot_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ichi_conv'], name="Senkou Span B", line=dict(color='yellow')), row=plot_row, col=1)
        elif indicator == "STD":
            fig.add_trace(go.Scatter(x=df.index, y=df['std'], name="Standart Sapma", line=dict(color='blue')), row=plot_row, col=1)
        plot_row += 1
    
    # Grafik düzenini güncelle
    fig.update_layout(
        height=300 * num_plots,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        title=f"{symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    return fig

# Fiyat kutusu fonksiyonu
def show_price_box(df, symbol):
    is_none = df is None
    is_empty = False if is_none else df.empty
    has_close = False if is_none or is_empty else 'Close' in df.columns
    close_col = None
    close_all_nan = False
    if has_close:
        close_col = df['Close']
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        close_all_nan = bool(close_col.isnull().all())
    if is_none or is_empty or not has_close or close_all_nan:
        st.error("Fiyat bilgisi bulunamadı veya 'Close' sütunu boş!")
        return
    # Son kapanış satırı
    last_row = df.iloc[-1]
    try:
        # Her durumda float'a zorla
        price = float(last_row['Close'].iloc[0]) if isinstance(last_row['Close'], pd.Series) else float(last_row['Close'])
        if pd.isna(price):
            st.error("Son fiyat (Close) değeri eksik!")
            return
        if len(df) > 1:
            prev_close_val = df['Close'].iloc[-2]
            prev_close = float(prev_close_val.iloc[0]) if isinstance(prev_close_val, pd.Series) else float(prev_close_val)
            if not pd.isna(prev_close):
                diff = price - prev_close
                pct = (diff / prev_close) * 100 if prev_close != 0 else 0
            else:
                diff = 0
                pct = 0
        else:
            diff = 0
            pct = 0
        color = "#16c784" if diff > 0 else ("#ea3943" if diff < 0 else "#cccccc")
        arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "→")
        # Geciken veri saati: şu an - 15 dakika
        gecikmeli_saat = (datetime.now() - timedelta(minutes=15)).strftime('%H:%M:%S')
        st.markdown(f"""
        <div style='background:#222;padding:2rem 1rem 1rem 1rem;border-radius:1rem;max-width:400px;margin-bottom:1rem;'>
            <div style='font-size:2rem;font-weight:bold;color:white;margin-bottom:0.5rem;'>{symbol}</div>
            <div style='font-size:3.5rem;font-weight:bold;color:{color};display:inline-block;'>{price:,.2f}</div>
            <span style='font-size:2rem;font-weight:bold;color:{color};margin-left:1rem;'>{arrow} {diff:+.2f} ({pct:+.2f}%)</span>
            <div style='font-size:1rem;color:#aaa;margin-top:0.5rem;'>🕒 Geciken Veriler · {gecikmeli_saat}</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Fiyat kutusu gösterilemiyor: {e}")

# Ana uygulama
if __name__ == "__main__":
    # Veri çek
    df = get_data(selected_symbol, period, timeframe)
    veri_cekilme_zamani = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # --- FİYAT KUTUSU ---
    show_price_box(df, selected_symbol)
    # --- FİYAT KUTUSU ---
    
    if df is not None and not df.empty:
        # İndikatörleri hesapla
        if selected_indicators:
            df = calculate_indicators(df, selected_indicators)
            
        # Grafiği çiz
        fig = plot_chart(df, selected_symbol, selected_indicators, show_price_chart=show_price_chart)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            
        # Veri çekilme zamanını göster
        st.info(f"Son veri çekilme zamanı: {veri_cekilme_zamani}")
        
        # Tarih sütununu normalize et (sadece tarih)
        df['date_only'] = df.index.date if hasattr(df.index, 'date') else pd.to_datetime(df.index).date
        # Seçilen tarihe ait veriyi bul
        day_data = df[df['date_only'] == selected_date]
        tablo = None
        if not day_data.empty:
            tavan = float(day_data['High'].max())
            dip = float(day_data['Low'].min())
            kapanis = float(day_data['Close'].iloc[-1])
            acilis = float(day_data['Open'].iloc[0])
            tablo = pd.DataFrame({
                'Açılış': [acilis],
                'Kapanış': [kapanis],
                'Tavan (Yüksek)': [tavan],
                'Dip (Düşük)': [dip],
                'Tarih': [selected_date.strftime('%Y-%m-%d')]
            })
        # Günlük fiyat tablosu ana ekranda gösterilecekse
        if show_price_table:
            st.subheader("Günlük Fiyat Seviyeleri")
            if tablo is not None:
                st.table(tablo)
            else:
                st.info(f"{selected_date.strftime('%Y-%m-%d')} için veri bulunamadı.")
        # Sinyalleri göster
        if selected_indicators:
            st.subheader("Sinyaller")
            last_row = df.iloc[-1]
            
            for indicator in selected_indicators:
                if indicator == "MA":
                    try:
                        close_val = float(last_row['Close'].iloc[0] if isinstance(last_row['Close'], pd.Series) else last_row['Close'])
                        ma20_val = float(last_row['ma20'].iloc[0] if isinstance(last_row['ma20'], pd.Series) else last_row['ma20'])
                        ma50_val = float(last_row['ma50'].iloc[0] if isinstance(last_row['ma50'], pd.Series) else last_row['ma50'])
                        ma200_val = float(last_row['ma200'].iloc[0] if isinstance(last_row['ma200'], pd.Series) else last_row['ma200'])
                        
                        if close_val > ma20_val and ma20_val > ma50_val and ma50_val > ma200_val:
                            st.success(f"MA: GÜÇLÜ AL (Fiyat: {close_val:.2f}, MA20: {ma20_val:.2f}, MA50: {ma50_val:.2f}, MA200: {ma200_val:.2f})")
                        elif close_val < ma20_val and ma20_val < ma50_val and ma50_val < ma200_val:
                            st.error(f"MA: GÜÇLÜ SAT (Fiyat: {close_val:.2f}, MA20: {ma20_val:.2f}, MA50: {ma50_val:.2f}, MA200: {ma200_val:.2f})")
                        else:
                            st.info(f"MA: BEKLE (Fiyat: {close_val:.2f}, MA20: {ma20_val:.2f}, MA50: {ma50_val:.2f}, MA200: {ma200_val:.2f})")
                    except (KeyError, ValueError, AttributeError):
                        st.warning("MA değerleri hesaplanamadı")
                
                elif indicator == "STOCH":
                    try:
                        stoch_k = float(last_row['stoch_k'].iloc[0] if isinstance(last_row['stoch_k'], pd.Series) else last_row['stoch_k'])
                        stoch_d = float(last_row['stoch_d'].iloc[0] if isinstance(last_row['stoch_d'], pd.Series) else last_row['stoch_d'])
                        
                        if stoch_k < 20 and stoch_d < 20:
                            st.success(f"Stokastik: GÜÇLÜ AL (K: {stoch_k:.2f}, D: {stoch_d:.2f})")
                        elif stoch_k > 80 and stoch_d > 80:
                            st.error(f"Stokastik: GÜÇLÜ SAT (K: {stoch_k:.2f}, D: {stoch_d:.2f})")
                        else:
                            st.info(f"Stokastik: BEKLE (K: {stoch_k:.2f}, D: {stoch_d:.2f})")
                    except (KeyError, ValueError, AttributeError):
                        st.warning("Stokastik değerleri hesaplanamadı")
                
                elif indicator == "FIB":
                    try:
                        close_val = float(last_row['Close'].iloc[0] if isinstance(last_row['Close'], pd.Series) else last_row['Close'])
                        fib_0 = float(last_row['fib_0'].iloc[0] if isinstance(last_row['fib_0'], pd.Series) else last_row['fib_0'])
                        fib_618 = float(last_row['fib_0.618'].iloc[0] if isinstance(last_row['fib_0.618'], pd.Series) else last_row['fib_0.618'])
                        
                        if close_val < fib_0:
                            st.success(f"Fibonacci: DESTEK SEVİYESİ - AL (Fiyat: {close_val:.2f}, Destek: {fib_0:.2f})")
                        elif close_val > fib_618:
                            st.error(f"Fibonacci: DİRENÇ SEVİYESİ - SAT (Fiyat: {close_val:.2f}, Direnç: {fib_618:.2f})")
                        else:
                            st.info(f"Fibonacci: BEKLE (Fiyat: {close_val:.2f}, Destek: {fib_0:.2f}, Direnç: {fib_618:.2f})")
                    except (KeyError, ValueError, AttributeError):
                        st.warning("Fibonacci değerleri hesaplanamadı")
                
                elif indicator == "ICHIMOKU":
                    try:
                        close_val = float(last_row['Close'].iloc[0] if isinstance(last_row['Close'], pd.Series) else last_row['Close'])
                        tenkan = float(last_row['ichi_a'].iloc[0] if isinstance(last_row['ichi_a'], pd.Series) else last_row['ichi_a'])
                        kijun = float(last_row['ichi_b'].iloc[0] if isinstance(last_row['ichi_b'], pd.Series) else last_row['ichi_b'])
                        senkou_a = float(last_row['ichi_base'].iloc[0] if isinstance(last_row['ichi_base'], pd.Series) else last_row['ichi_base'])
                        senkou_b = float(last_row['ichi_conv'].iloc[0] if isinstance(last_row['ichi_conv'], pd.Series) else last_row['ichi_conv'])
                        
                        if close_val > senkou_a and close_val > senkou_b and tenkan > kijun:
                            st.success(f"Ichimoku: GÜÇLÜ AL (Fiyat: {close_val:.2f}, Tenkan: {tenkan:.2f}, Kijun: {kijun:.2f})")
                        elif close_val < senkou_a and close_val < senkou_b and tenkan < kijun:
                            st.error(f"Ichimoku: GÜÇLÜ SAT (Fiyat: {close_val:.2f}, Tenkan: {tenkan:.2f}, Kijun: {kijun:.2f})")
                        else:
                            st.info(f"Ichimoku: BEKLE (Fiyat: {close_val:.2f}, Tenkan: {tenkan:.2f}, Kijun: {kijun:.2f})")
                    except (KeyError, ValueError, AttributeError):
                        st.warning("Ichimoku değerleri hesaplanamadı")
                
                elif indicator == "STD":
                    try:
                        close_val = float(last_row['Close'].iloc[0] if isinstance(last_row['Close'], pd.Series) else last_row['Close'])
                        std_val = float(last_row['std'].iloc[0] if isinstance(last_row['std'], pd.Series) else last_row['std'])
                        ma20_val = float(last_row['ma20'].iloc[0] if isinstance(last_row['ma20'], pd.Series) else last_row['ma20'])
                        
                        if std_val > ma20_val * 0.02:  # Volatilite yüksek
                            st.warning(f"Standart Sapma: YÜKSEK VOLATİLİTE (Std: {std_val:.2f}, MA20: {ma20_val:.2f})")
                        else:
                            st.info(f"Standart Sapma: NORMAL VOLATİLİTE (Std: {std_val:.2f}, MA20: {ma20_val:.2f})")
                    except (KeyError, ValueError, AttributeError):
                        st.warning("Standart Sapma değerleri hesaplanamadı")
                
                elif indicator == "RSI":
                    try:
                        rsi_val = float(last_row['rsi'].iloc[0] if isinstance(last_row['rsi'], pd.Series) else last_row['rsi'])
                        if rsi_val < 30:
                            st.success(f"RSI: GÜÇLÜ AL ({rsi_val:.2f})")
                        elif rsi_val > 70:
                            st.error(f"RSI: GÜÇLÜ SAT ({rsi_val:.2f})")
                        else:
                            st.info(f"RSI: BEKLE ({rsi_val:.2f})")
                    except (KeyError, ValueError, AttributeError):
                        st.warning("RSI değeri hesaplanamadı")
                
                elif indicator == "MACD":
                    try:
                        macd_val = float(last_row['macd'].iloc[0] if isinstance(last_row['macd'], pd.Series) else last_row['macd'])
                        macd_signal_val = float(last_row['macd_signal'].iloc[0] if isinstance(last_row['macd_signal'], pd.Series) else last_row['macd_signal'])
                        
                        if macd_val > macd_signal_val:
                            st.success(f"MACD: AL (MACD: {macd_val:.2f}, Signal: {macd_signal_val:.2f})")
                        elif macd_val < macd_signal_val:
                            st.error(f"MACD: SAT (MACD: {macd_val:.2f}, Signal: {macd_signal_val:.2f})")
                        else:
                            st.info(f"MACD: BEKLE (MACD: {macd_val:.2f}, Signal: {macd_signal_val:.2f})")
                    except (KeyError, ValueError, AttributeError):
                        st.warning("MACD değerleri hesaplanamadı")
                
                elif indicator == "BB":
                    try:
                        close_val = float(last_row['Close'].iloc[0] if isinstance(last_row['Close'], pd.Series) else last_row['Close'])
                        bb_low_val = float(last_row['bb_low'].iloc[0] if isinstance(last_row['bb_low'], pd.Series) else last_row['bb_low'])
                        bb_high_val = float(last_row['bb_high'].iloc[0] if isinstance(last_row['bb_high'], pd.Series) else last_row['bb_high'])
                        bb_mid_val = float(last_row['bb_mid'].iloc[0] if isinstance(last_row['bb_mid'], pd.Series) else last_row['bb_mid'])
                        
                        if close_val < bb_low_val:
                            st.success(f"Bollinger: ALT BAND - AL (Fiyat: {close_val:.2f}, Alt Bant: {bb_low_val:.2f})")
                        elif close_val > bb_high_val:
                            st.error(f"Bollinger: ÜST BAND - SAT (Fiyat: {close_val:.2f}, Üst Bant: {bb_high_val:.2f})")
                        else:
                            st.info(f"Bollinger: BEKLE (Fiyat: {close_val:.2f}, Orta Bant: {bb_mid_val:.2f})")
                    except (KeyError, ValueError, AttributeError):
                        st.warning("Bollinger Bands değerleri hesaplanamadı")
        else:
            st.warning("Lütfen en az bir indikatör seçin!")
    else:
        st.error("Veri çekilemedi!")
