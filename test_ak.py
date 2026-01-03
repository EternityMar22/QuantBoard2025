import akshare as ak
import pandas as pd

print("Start AKShare Download")
try:
    # Test 515450
    df = ak.fund_etf_hist_em(
        symbol="515450",
        period="daily",
        start_date="20240101",
        end_date="20240110",
        adjust="qfq",
    )
    print("Download Complete")
    print(df.head())
except Exception as e:
    print(f"Error: {e}")
