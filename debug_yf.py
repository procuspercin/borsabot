import yfinance as yf
import pandas as pd

symbol = "AKBNK.IS"
period = "1y"
interval = "1d"

print(f"Downloading {symbol}...")
try:
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    print("Download complete.")
    print(f"Shape: {df.shape}")
    if not df.empty:
        print("Head:")
        print(df.head())
        print("Columns:", df.columns)
        print("Index:", df.index)
    else:
        print("Dataframe is empty!")
except Exception as e:
    print(f"Error: {e}")
