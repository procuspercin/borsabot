"""
Grafikler. Figürler Python'da kurulur, JSON olarak tarayıcıya gönderilir ve
plotly.js tarafından çizilir; sunucu görüntü üretmez.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_price_chart(df: pd.DataFrame, indicators: list):
    """
    Detay sayfasının mum grafiği + indikatör alt panelleri.
    app.py'deki render_detail içinden birebir taşındı; tek farkı figürü
    çizmek yerine döndürmesi (tarayıcı tarafında plotly.js çiziyor).
    """
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

    return fig


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
