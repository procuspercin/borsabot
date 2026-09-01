"""
BorsaBot Finans — web katmanı (FastAPI + Jinja + htmx).

Streamlit'in yerini alır. Tüm veri/analiz mantığı core/market.py içinde;
burada yalnızca yönlendirme ve şablon doldurma var. Sayfa etkileşimleri
tam sayfa yerine yalnızca ilgili parçayı günceller.
"""

from __future__ import annotations

import io
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.io as pio
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core import charts, daily_log as dl, market as m
from web import session as sess

BASE_DIR = m.PROJECT_ROOT / "web"

app = FastAPI(title="BorsaBot Finans", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "templates")
templates.env.globals.update(
    fmt_price=m.fmt_price,
    trend_class=m.trend_class,
    trend_arrow=m.trend_arrow,
    sparkline_svg=m.sparkline_svg,
    market_card_html=m.market_card_html,
    quote_row_html=m.quote_row_html,
    signal_badge=m.signal_badge,
    forecast_card_html=m._forecast_card_html,
    safe_link=m.safe_link,
    MARKET_TABS=m.MARKET_TABS,
    SECTOR_INDICES=m.SECTOR_INDICES,
    TICKER_STRIP=m.TICKER_STRIP,
    BIST100_SYMBOLS=m.BIST100_SYMBOLS,
    HORIZON_LABELS=m.HORIZON_LABELS,
)

INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d", "1wk", "1mo"]
PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
ALL_INDICATORS = ["MA20", "MA50", "MA200", "RSI", "MACD", "Bollinger", "Stoch", "Ichimoku", "CCI"]
MONTHS_TR = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
    7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
}


# --------------------------------------------------------------------------- #
# Oturum
# --------------------------------------------------------------------------- #

def _session(request: Request) -> tuple[str, dict]:
    return sess.get(request.cookies.get(sess.COOKIE_NAME))


def _render(request: Request, template: str, ctx: dict, sid: str) -> HTMLResponse:
    response = templates.TemplateResponse(request, template, ctx)
    response.set_cookie(
        sess.COOKIE_NAME, sid,
        max_age=sess.SESSION_TTL, httponly=True, samesite="lax",
    )
    return response


def _home_quotes(state: dict) -> dict:
    """
    Ana sayfa fiyatları. Önbellek anahtarı sembol demetine bağlı olduğu için
    her çağrıda AYNI demet üretilmeli; aksi halde parça güncellemeleri
    önbelleği ıskalayıp yeniden ağ isteği atıyor.
    """
    extra = tuple(state["watchlist"]) + tuple(state["last_viewed"][:6])
    return m.prefetch_home_quotes(extra)


def _normalize(symbol: str) -> str:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return ""
    if "." in symbol or "=" in symbol or "-" in symbol:
        return symbol
    return f"{symbol}.IS"


# --------------------------------------------------------------------------- #
# Ana sayfa
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    sid, state = _session(request)
    quotes = _home_quotes(state)
    movers = m.get_market_movers(quotes)
    n_side = max(1, min(6, len(movers) // 2))
    return _render(request, "home.html", {
        "quotes": quotes,
        "state": state,
        "gainers": movers.head(n_side).to_dict("records"),
        "losers": movers.tail(n_side).iloc[::-1].to_dict("records"),
        "news": m.get_bloomberg_news()[:6],
        "ai": _ai_context(state, state["watchlist"][0] if state["watchlist"] else "THYAO.IS"),
    }, sid)


@app.get("/p/piyasa/{tab}", response_class=HTMLResponse)
def market_tab(request: Request, tab: str):
    """Piyasa sekmesi içeriği — veri zaten önbellekte, tek parça güncellenir."""
    sid, state = _session(request)
    items = m.MARKET_TABS.get(tab)
    if items is None:
        items = next(iter(m.MARKET_TABS.values()))
    quotes = _home_quotes(state)
    return _render(request, "partials/market_cards.html",
                   {"items": items, "quotes": quotes, "active": tab}, sid)


# --------------------------------------------------------------------------- #
# İzleme listesi
# --------------------------------------------------------------------------- #

@app.post("/p/izleme/{symbol}", response_class=HTMLResponse)
def watch_toggle(request: Request, symbol: str):
    sid, state = _session(request)
    sess.toggle_watch(state, _normalize(symbol))
    quotes = _home_quotes(state)
    return _render(request, "partials/side_lists.html", {"state": state, "quotes": quotes}, sid)


@app.post("/p/izleme-ekle", response_class=HTMLResponse)
def watch_add(request: Request, symbol: str = Form("")):
    sid, state = _session(request)
    sym = _normalize(symbol)
    if sym and sym not in state["watchlist"]:
        state["watchlist"].insert(0, sym)
    quotes = _home_quotes(state)
    return _render(request, "partials/side_lists.html", {"state": state, "quotes": quotes}, sid)


# --------------------------------------------------------------------------- #
# Teknik özet penceresi
# --------------------------------------------------------------------------- #

@app.get("/p/ozet/{symbol}", response_class=HTMLResponse)
def popup(request: Request, symbol: str):
    sid, state = _session(request)
    sym = _normalize(symbol)
    ctx = m.get_analysis_context(sym)
    if ctx is None:
        return HTMLResponse('<div class="gf-modal-bg" onclick="closeModal()">'
                            '<div class="gf-modal">Bu sembol için veri alınamadı.</div></div>')
    return _render(request, "partials/popup.html", {
        "symbol": sym, "ctx": ctx,
        "signals": m._signals_df_to_dict(ctx["signals"]),
        "in_watchlist": sym in state["watchlist"],
    }, sid)


@app.get("/p/kapat", response_class=HTMLResponse)
def modal_close():
    return HTMLResponse("")


# --------------------------------------------------------------------------- #
# Hisse detayı
# --------------------------------------------------------------------------- #

def _futures_map(symbol: str) -> dict:
    now = datetime.now()
    out = {"Spot (Hisse)": symbol}
    for i in range(3):
        month, year = now.month + i, now.year
        if month > 12:
            month, year = month - 12, year + 1
        out[f"{MONTHS_TR[month]} {year} Vade"] = f"F_{symbol.split('.')[0]}{month:02d}{str(year)[-2:]}.IS"
    return out


def _fix_interval(period: str, interval: str) -> tuple[str, str | None]:
    """Geçersiz periyot/aralık kombinasyonlarını düzeltir; uyarı metni döndürür."""
    original = interval
    if period in ("1mo", "3mo") and interval == "1m":
        interval = "1h"
    elif period in ("6mo", "1y", "2y") and interval in ("1m", "5m", "15m"):
        interval = "1d"
    elif period in ("5y", "10y", "max") and interval in ("1m", "5m", "15m", "1h", "4h"):
        interval = "1d"
    if original != interval:
        return interval, (f"{period} periyodu için {original} verisi mevcut değil. "
                          f"Otomatik olarak {interval} seçildi.")
    return interval, None


@app.get("/hisse/{symbol}", response_class=HTMLResponse)
def detail(request: Request, symbol: str,
           period: str = "1y", interval: str = "1d",
           ind: list[str] = Query(default=["MA20", "MA50"]),   # boş liste de geçerli
           vade: str = "Spot (Hisse)"):
    sid, state = _session(request)
    sym = _normalize(symbol)
    sess.remember_symbol(state, sym)

    # Form her zaman boş bir "ind" gönderiyor; böylece hiç indikatör seçilmediğinde
    # FastAPI varsayılana dönmüyor ve sade fiyat grafiği görülebiliyor.
    ind = [i for i in ind if i]
    interval, notice = _fix_interval(period, interval)
    futures = _futures_map(sym)
    target = futures.get(vade, sym)

    quote = m.get_quotes((sym,)).get(sym)
    df = m.get_stock_data(target, period, interval)
    stats = None
    if df is not None and not df.empty:
        last = df.iloc[-1]

        def val(v):
            return float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)

        stats = {k: val(last[k]) for k in ("Close", "Open", "High", "Low")}

    signals = None
    if df is not None and not df.empty:
        signals = m._signals_df_to_dict(m.calculate_technical_signals(df))

    return _render(request, "detail.html", {
        "symbol": sym, "target": target, "quote": quote, "stats": stats, "signals": signals,
        "period": period, "interval": interval, "indicators": ind, "vade": vade,
        "futures": list(futures), "notice": notice,
        "intervals": INTERVALS, "periods": PERIODS, "all_indicators": ALL_INDICATORS,
        "has_data": df is not None and not df.empty,
        "state": state,
        "ai": _ai_context(state, sym),
    }, sid)


@app.get("/api/grafik")
def chart_json(symbol: str, period: str = "1y", interval: str = "1d",
               ind: list[str] = Query(default=["MA20", "MA50"])):
    """Mum grafiğini plotly JSON'u olarak döndürür; çizimi tarayıcı yapar."""
    ind = [i for i in ind if i]
    interval, _ = _fix_interval(period, interval)
    df = m.get_stock_data(_normalize(symbol), period, interval)
    if df is None or df.empty:
        return JSONResponse({"error": "Veri bulunamadı."}, status_code=404)
    return JSONResponse(pio.to_json(charts.build_price_chart(df, ind), remove_uids=True),
                        media_type="application/json")


@app.get("/indir/{symbol}.csv")
def download_csv(symbol: str, period: str = "1y", interval: str = "1d"):
    sym = _normalize(symbol)
    interval, _ = _fix_interval(period, interval)
    df = m.get_stock_data(sym, period, interval)
    if df is None or df.empty:
        return JSONResponse({"error": "Veri bulunamadı."}, status_code=404)
    buf = io.StringIO()
    df.to_csv(buf)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{sym.replace(".IS", "")}_{period}.csv"'},
    )


# --------------------------------------------------------------------------- #
# Asistan
# --------------------------------------------------------------------------- #

QUICK_QUESTIONS = [
    "{s} teknik olarak nasıl görünüyor?",
    "RSI ve MACD şu an ne gösteriyor?",
    "Hangi seviyeler destek, hangileri direnç?",
    "Geri çekilmede hangi bandlar izlenir?",
]


def _ai_context(state: dict, symbol: str) -> dict:
    short = symbol.replace(".IS", "")
    return {
        "symbol": symbol,
        "short": short,
        "history": sess.chat(state, symbol),
        "quick": [q.format(s=short) for q in QUICK_QUESTIONS],
        "has_key": bool(m.get_gemini_key()),
        "usage": m.gemini_usage(),
    }


@app.post("/p/asistan/{symbol}", response_class=HTMLResponse)
def assistant(request: Request, symbol: str, soru: str = Form(...)):
    sid, state = _session(request)
    sym = _normalize(symbol)
    history = sess.chat(state, sym)
    history.append({"role": "user", "content": soru})

    ctx = m.get_analysis_context(sym)
    if ctx is None:
        history.append({"role": "assistant", "content": "⚠️ Bu sembol için teknik veri alınamadı."})
    else:
        context_text = ctx["text"]
        for extra in m.detect_symbols(soru, exclude=sym)[:2]:
            extra_ctx = m.get_analysis_context(extra)
            if extra_ctx:
                context_text += "\n\n" + extra_ctx["text"]
        answer, err = m.ask_gemini(history, context_text)
        history.append({"role": "assistant", "content": answer or f"⚠️ {err}"})

    return _render(request, "partials/ai_panel.html", {"ai": _ai_context(state, sym)}, sid)


@app.post("/p/asistan/{symbol}/temizle", response_class=HTMLResponse)
def assistant_clear(request: Request, symbol: str):
    sid, state = _session(request)
    sym = _normalize(symbol)
    state["chats"][sym] = []
    return _render(request, "partials/ai_panel.html", {"ai": _ai_context(state, sym)}, sid)


# --------------------------------------------------------------------------- #
# ML tahmin
# --------------------------------------------------------------------------- #

@app.get("/ml-tahmin", response_class=HTMLResponse)
def forecast(request: Request, hisse: str = ""):
    sid, state = _session(request)
    tickers = m.ml_supported_tickers()
    data = err = None
    chart = None
    if hisse and tickers:
        data, err = m.ml_forecast(hisse)
        if data:
            chart = pio.to_json(charts._forecast_chart(data), remove_uids=True)
    return _render(request, "forecast.html", {
        "tickers": tickers, "selected": hisse or (tickers[0] if tickers else ""),
        "data": data, "err": err, "chart": chart,
        "available": m.forecaster_available(), "state": state,
    }, sid)


# --------------------------------------------------------------------------- #
# Günlük kayıt
# --------------------------------------------------------------------------- #

@app.get("/gunluk-kayit", response_class=HTMLResponse)
def daily_log_page(request: Request, tarih: str = "", hisse: str = "", ara: str = ""):
    sid, state = _session(request)
    symbols = dl.get_target_symbols()
    # Son 1 ay arka planda tamamlanır; kullanıcının elle bir şey yapması gerekmez.
    m.ensure_last_month(symbols)

    now = dl.now_istanbul()
    df = dl.fetch_all()
    has_rows = df is not None and not df.empty

    per_day: dict[str, dict] = {}
    if has_rows:
        for date_str, group in df.groupby(df["Tarih"].astype(str)):
            signals = group["Genel Sinyal"].astype(str) if "Genel Sinyal" in group else pd.Series(dtype=str)
            per_day[date_str] = {
                "count": len(group),
                "al": int(signals.str.contains("AL").sum()),
                "sat": int(signals.str.contains("SAT").sum()),
            }

    # Varsayılan gün: veri olan en yeni gün
    if not tarih and per_day:
        tarih = max(per_day)

    table = df
    if has_rows:
        if tarih:
            table = table[table["Tarih"].astype(str) == tarih]
        if hisse:
            table = table[table["Hisse"].astype(str) == hisse]
        if ara:
            mask = table.apply(lambda r: ara.lower() in " ".join(map(str, r.values)).lower(), axis=1)
            table = table[mask]

    return _render(request, "daily_log.html", {
        "symbols": symbols, "now": now, "today": now.date().isoformat(),
        "weeks": _calendar_weeks(now.date(), per_day, m.BACKFILL_DAYS),
        "per_day": per_day,
        "rows": table.to_dict("records") if has_rows and not table.empty else [],
        "columns": list(df.columns) if has_rows else [],
        "syms": dl.distinct_symbols(),
        "tarih": tarih, "hisse": hisse, "ara": ara,
        "total": dl.total_record_count(),
        "backfill": m.backfill_status(),
        "state": state,
    }, sid)


def _calendar_weeks(today: date, per_day: dict, days: int) -> list[list[dict]]:
    """
    Son `days` günü kapsayan, pazartesiyle başlayan hafta satırları üretir.
    Her hücre: tarih, o güne ait kayıt özeti, bugün mü, aralık dışında mı.
    """
    first = today - timedelta(days=days)
    first -= timedelta(days=first.weekday())          # haftanın başına hizala
    weeks, cursor = [], first
    while cursor <= today:
        week = []
        for _ in range(7):
            iso = cursor.isoformat()
            week.append({
                "date": cursor, "iso": iso, "day": cursor.day,
                "stats": per_day.get(iso),
                "is_today": cursor == today,
                "future": cursor > today,
                "weekend": cursor.weekday() >= 5,
            })
            cursor += timedelta(days=1)
        weeks.append(week)
    return weeks


@app.post("/gunluk-kayit/yenile")
def daily_log_refresh(request: Request):
    m.ensure_last_month(dl.get_target_symbols(), force=True)
    return RedirectResponse("/gunluk-kayit", status_code=303)


@app.get("/p/benzer/{ticker}/{horizon}", response_class=HTMLResponse)
def similar_events(request: Request, ticker: str, horizon: int, p: float = 0.5):
    """Modelin bugünkü olasılığına en yakın 5 geçmiş örnek ve sonraki seyirleri."""
    sid, state = _session(request)
    events = m.similar_past_events(ticker, horizon, p, limit=5)
    chart = pio.to_json(charts.similar_events_chart(events, horizon), remove_uids=True) if events else None
    return _render(request, "partials/similar.html", {
        "ticker": ticker.upper().replace(".IS", ""), "horizon": horizon,
        "probability": p, "events": events, "chart": chart,
        "label": m.HORIZON_LABELS.get(horizon, f"{horizon} gün"),
    }, sid)


@app.get("/haberler", response_class=HTMLResponse)
def news_page(request: Request):
    sid, state = _session(request)
    return _render(request, "news.html", {"news": m.get_bloomberg_news(), "state": state}, sid)


@app.get("/arama")
def search(request: Request, sembol: str = ""):
    sym = _normalize(sembol)
    return RedirectResponse(f"/hisse/{sym}" if sym else "/", status_code=303)
