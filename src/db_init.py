import duckdb
from pathlib import Path

DB_PATH = Path("quant.db")


def init_db() -> None:
    """Initialize DuckDB database and create tables if they don't exist."""

    # Use context manager for database connection
    with duckdb.connect(str(DB_PATH)) as con:
        # 1. market_data: OHLCV
        con.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                date DATE,
                ticker VARCHAR,
                market VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (date, ticker)
            )
        """)
        print("Table 'market_data' check/create: Done")

        # 2. portfolio_snapshot: Current holdings
        con.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshot (
                ticker VARCHAR,
                qty DOUBLE,
                market VARCHAR,
                cost_basis DOUBLE
            )
        """)
        print("Table 'portfolio_snapshot' check/create: Done")

        # 3. fx_rates: Exchange rates
        con.execute("""
            CREATE TABLE IF NOT EXISTS fx_rates (
                date DATE,
                pair VARCHAR,
                rate DOUBLE,
                PRIMARY KEY (date, pair)
            )
        """)
        print("Table 'fx_rates' check/create: Done")


if __name__ == "__main__":
    init_db()
