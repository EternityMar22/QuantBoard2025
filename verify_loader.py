import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
from src.data_loader import sync_data, sync_fx_rates
import duckdb


def verify():
    print("=== Starting Data Loader Verification ===")

    # 1. Sync Market Data
    tickers = ["000001", "AAPL"]
    print(f"\n>> Syncing data for: {tickers}")
    sync_data(tickers)

    # 2. Sync FX Rates
    print("\n>> Syncing FX Rates")
    sync_fx_rates()

    # 3. Check DB
    print("\n>> Verifying DB content")
    with duckdb.connect("quant.db") as con:
        count_market = con.execute("SELECT count(*) FROM market_data").fetchone()[0]
        count_fx = con.execute("SELECT count(*) FROM fx_rates").fetchone()[0]

        print(f"Rows in market_data: {count_market}")
        print(f"Rows in fx_rates: {count_fx}")

        # Sample data
        print("\nSample Market Data:")
        print(con.execute("SELECT * FROM market_data LIMIT 3").pl())

        print("\nSample FX Data:")
        print(con.execute("SELECT * FROM fx_rates LIMIT 3").pl())


if __name__ == "__main__":
    verify()
