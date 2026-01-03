import yfinance as yf

print("Start YF Download")
try:
    df = yf.download(["QQQ", "SPY"], period="5d", progress=False)
    print("Download Complete")
    print(df.head())
except Exception as e:
    print(f"Error: {e}")
