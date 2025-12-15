import akshare as ak
import yfinance as yf
import polars as pl
import pandas as pd
import duckdb
from datetime import datetime, date
from typing import List, Optional

DB_PATH = "quant.db"


def fetch_ashare(symbol: str) -> pl.DataFrame:
    """
    Fetch A-share daily data using Akshare.
    Args:
        symbol: A-share stock symbol (e.g., '000001').
    Returns:
        pl.DataFrame: Columns [date, ticker, market, open, high, low, close, volume]
    """
    try:
        # Fetch data (defaulting to recent years for efficiency, or full history)
        # using qfq (forward adjusted) by default
        df_raw = ak.stock_zh_a_hist(
            symbol=symbol, period="daily", start_date="20200101", adjust="qfq"
        )

        if df_raw is None or df_raw.empty:
            print(f"Warning: No data found for A-share {symbol}")
            return pl.DataFrame()

        # Rename columns to standardized English names
        # Akshare typical columns: 日期, 开盘, 收盘, 最高, 最低, 成交量, ...
        q = pl.from_pandas(df_raw).select(
            [
                pl.col("日期").cast(pl.Date).alias("date"),
                pl.lit(symbol).alias("ticker"),
                pl.lit("CN").alias("market"),
                pl.col("开盘").cast(pl.Float64).alias("open"),
                pl.col("最高").cast(pl.Float64).alias("high"),
                pl.col("最低").cast(pl.Float64).alias("low"),
                pl.col("收盘").cast(pl.Float64).alias("close"),
                pl.col("成交量").cast(pl.Float64).alias("volume"),
            ]
        )
        return q

    except Exception as e:
        print(f"Error fetching A-share {symbol}: {e}")
        return pl.DataFrame()


def fetch_usstock(symbol: str) -> pl.DataFrame:
    """
    Fetch US stock daily data using yfinance.
    Args:
        symbol: US stock ticker (e.g., 'AAPL').
    Returns:
        pl.DataFrame: Columns [date, ticker, market, open, high, low, close, volume]
    """
    try:
        # Fetch data
        # auto_adjust=True returns Close as adjusted close
        # returning simple dataframe
        df_pd = yf.download(
            symbol, start="2020-01-01", progress=False, auto_adjust=True
        )

        if df_pd.empty:
            print(f"Warning: No data found for US stock {symbol}")
            return pl.DataFrame()

        # Validating index and columns
        # yfinance index is DatetimeIndex usually
        df_pd = df_pd.reset_index()

        # Determine column mapping based on what yfinance returns (generic or multi-index)
        # Fix for yfinance returning MultiIndex columns (Price, Ticker)
        if isinstance(df_pd.columns, pd.MultiIndex):
            # Collapse multi-index columns if present, taking the first level (Price)
            # But wait, if we download single ticker, it might be (Price, Ticker) or just Price
            # Usually single ticker download with auto_adjust=True gives plain columns if flattened?
            # Actually yf 0.2+ returns MultiIndex even for single ticker sometimes
            df_pd.columns = df_pd.columns.get_level_values(0)

        q = pl.from_pandas(df_pd).select(
            [
                pl.col("Date").cast(pl.Date).alias("date"),
                pl.lit(symbol).alias("ticker"),
                pl.lit("US").alias("market"),
                pl.col("Open").cast(pl.Float64).alias("open"),
                pl.col("High").cast(pl.Float64).alias("high"),
                pl.col("Low").cast(pl.Float64).alias("low"),
                pl.col("Close").cast(pl.Float64).alias("close"),
                pl.col("Volume").cast(pl.Float64).alias("volume"),
            ]
        )
        return q

    except Exception as e:
        print(f"Error fetching US stock {symbol}: {e}")
        return pl.DataFrame()


def sync_data(tickers: List[str]) -> None:
    """
    Fetch data for list of tickers and sync to DuckDB market_data table.
    """
    all_data = []

    for t in tickers:
        # Simple heuristic: All digits -> A-share (CN), else US
        if t.isdigit():
            print(f"Fetching A-share: {t}")
            df = fetch_ashare(t)
        else:
            print(f"Fetching US stock: {t}")
            df = fetch_usstock(t)

        if not df.is_empty():
            all_data.append(df)

    if not all_data:
        print("No data fetched.")
        return

    # Concat all
    try:
        combined_df = pl.concat(all_data)

        print(f"Syncing {combined_df.height} rows to DuckDB...")

        # Upsert to DuckDB
        with duckdb.connect(DB_PATH) as con:
            con.execute("INSERT OR REPLACE INTO market_data SELECT * FROM combined_df")

        print("Sync completed.")
    except Exception as e:
        print(f"Error in sync_data: {e}")


def sync_fx_rates() -> None:
    """
    Fetch USD/CNY exchange rate and sync to fx_rates table.
    Using yfinance 'CNY=X'
    """
    try:
        print("Fetching USD/CNY rates...")
        df_pd = yf.download(
            "CNY=X", start="2020-01-01", progress=False, auto_adjust=True
        )

        if df_pd.empty:
            print("Warning: No FX data found.")
            return

        df_pd = df_pd.reset_index()

        if isinstance(df_pd.columns, pd.MultiIndex):
            df_pd.columns = df_pd.columns.get_level_values(0)

        # Create Polars DataFrame
        q = pl.from_pandas(df_pd).select(
            [
                pl.col("Date").cast(pl.Date).alias("date"),
                pl.lit("USD/CNY").alias("pair"),
                pl.col("Close").cast(pl.Float64).alias("rate"),
            ]
        )

        with duckdb.connect(DB_PATH) as con:
            con.execute("INSERT OR REPLACE INTO fx_rates SELECT * FROM q")

        print(f"Synced {q.height} FX rate records.")

    except Exception as e:
        print(f"Error syncing FX rates: {e}")


if __name__ == "__main__":
    # Test run
    # sync_data(["000001", "AAPL"])
    # sync_fx_rates()
    pass
