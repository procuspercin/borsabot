import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, symbol, period="3mo", interval="1d"):
        self.symbol = symbol
        self.interval = interval
        
        # Zaman aralığına göre period ayarla
        if interval == "15m":
            self.period = "5d"  # 15 dakikalık veriler için 5 gün
        elif interval == "1h":
            self.period = "1mo"  # Saatlik veriler için 1 ay
        elif interval == "4h":
            self.period = "3mo"  # 4 saatlik veriler için 3 ay
        else:  # 1d
            self.period = "1y"  # Günlük veriler için 1 yıl
        
    def get_data(self):
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=self.period, interval=self.interval)
            return df
        except Exception as e:
            print(f"Veri çekme hatası: {e}")
            return None