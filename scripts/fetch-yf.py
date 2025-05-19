import yfinance as yf
import pandas as pd

symbol = "PHPJPY=X"
data = yf.download(symbol, start="2014-03-01", end="2025-03-31", interval="1d")
print(data.head())
data.to_csv(f"./{symbol}.csv", index=True, encoding="utf-8-sig")