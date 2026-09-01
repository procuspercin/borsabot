"""
Oturum durumu: izleme listesi, son bakılanlar ve asistan sohbeti.

Streamlit'in st.session_state'inin yerini alır. Uygulama nginx arkasında tek
kullanıcılı çalıştığı için durum süreç içinde tutulur; tarayıcı yalnızca bir
oturum kimliği çerezi taşır.
"""

from __future__ import annotations

import threading
import time
import uuid

COOKIE_NAME = "bb_session"
SESSION_TTL = 30 * 24 * 3600          # 30 gün
WATCHLIST_DEFAULT = ["THYAO.IS", "ASELS.IS", "SISE.IS", "GARAN.IS"]

_lock = threading.Lock()
_sessions: dict[str, dict] = {}


def _new_state() -> dict:
    return {
        "touched": time.time(),
        "watchlist": list(WATCHLIST_DEFAULT),
        "last_viewed": [],
        "chats": {},                   # {sembol: [{role, content}, ...]}
    }


def _prune(now: float) -> None:
    for sid in [s for s, st in _sessions.items() if now - st["touched"] > SESSION_TTL]:
        _sessions.pop(sid, None)


def get(session_id: str | None) -> tuple[str, dict]:
    """Oturumu döndürür; yoksa yenisini açar. (oturum_id, durum) verir."""
    now = time.time()
    with _lock:
        _prune(now)
        if session_id and session_id in _sessions:
            state = _sessions[session_id]
            state["touched"] = now
            return session_id, state
        sid = uuid.uuid4().hex
        _sessions[sid] = _new_state()
        return sid, _sessions[sid]


def remember_symbol(state: dict, symbol: str) -> None:
    lv = [s for s in state["last_viewed"] if s != symbol]
    lv.insert(0, symbol)
    state["last_viewed"] = lv[:12]


def toggle_watch(state: dict, symbol: str) -> bool:
    """İzleme listesine ekler/çıkarır. Sonuçta listede mi, onu döndürür."""
    wl = state["watchlist"]
    if symbol in wl:
        wl.remove(symbol)
        return False
    wl.insert(0, symbol)
    return True


def chat(state: dict, symbol: str) -> list:
    return state["chats"].setdefault(symbol, [])
