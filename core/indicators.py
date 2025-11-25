import pandas as pd
import pandas_ta as ta
import numpy as np

class Indicators:
    @staticmethod
    def calculate_selected(df, selected_indicators):
        for indicator in selected_indicators:
            if indicator == "MA":
                # Hareketli Ortalamalar (20, 50, 200 günlük)
                df['ma20'] = df.ta.sma(length=20)
                df['ma50'] = df.ta.sma(length=50)
                df['ma200'] = df.ta.sma(length=200)
                
            elif indicator == "MACD":
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_diff'] = macd['MACDh_12_26_9']
                
            elif indicator == "BB":
                # Bollinger Bantları hesaplaması
                bb = df.ta.bbands(length=20, std=2)
                df['bb_high'] = bb['BBU_20_2.0']  # Üst bant
                df['bb_mid'] = bb['BBM_20_2.0']   # Orta bant
                df['bb_low'] = bb['BBL_20_2.0']   # Alt bant
                
            elif indicator == "RSI":
                df['rsi'] = df.ta.rsi(length=14)
                
            elif indicator == "STOCH":
                stoch = df.ta.stoch(high='High', low='Low', close='Close')
                df['stoch_k'] = stoch['STOCHk_14_3_3']
                df['stoch_d'] = stoch['STOCHd_14_3_3']
                
            elif indicator == "FIB":
                # Fibonacci Düzeltmesi için yüksek ve düşük noktaları bul
                high = df['High'].rolling(window=20).max()
                low = df['Low'].rolling(window=20).min()
                diff = high - low
                
                # Fibonacci seviyeleri
                df['fib_0'] = low
                df['fib_0.236'] = low + 0.236 * diff
                df['fib_0.382'] = low + 0.382 * diff
                df['fib_0.5'] = low + 0.5 * diff
                df['fib_0.618'] = low + 0.618 * diff
                df['fib_0.786'] = low + 0.786 * diff
                df['fib_1'] = high
                
            elif indicator == "ICHIMOKU":
                ichi = df.ta.ichimoku()
                if isinstance(ichi, tuple):
                    ichi = ichi[0]
                df['ichi_a'] = ichi['ISA_9']
                df['ichi_b'] = ichi['ISB_26']
                df['ichi_base'] = ichi['ITS_9']
                df['ichi_conv'] = ichi['IKS_26']
                
            elif indicator == "ATR":
                df['atr'] = df.ta.atr(length=14)
                
            elif indicator == "STD":
                # 20 günlük standart sapma
                df['std'] = df['Close'].rolling(window=20).std()
                
            elif indicator == "ADX":
                adx = df.ta.adx()
                df['adx'] = adx['ADX_14']
                df['di_plus'] = adx['DMP_14']
                df['di_minus'] = adx['DMN_14']
        
        return df
