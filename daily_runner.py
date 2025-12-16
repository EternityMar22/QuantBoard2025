import os
import sys
import requests
import polars as pl
from datetime import date
from dotenv import load_dotenv

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Load environment variables
load_dotenv()

from src.config import TICKERS, STRATEGY_REGISTRY, TG_TOKEN
from src.data_loader import sync_data, sync_fx_rates
from src.engine import BacktestEngine
from src.strategies.demo_strategy import simple_ma_strategy

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def ensure_strategies_registered():
    """
    Ensure at least one strategy is in the registry.
    If empty, register the default demo strategy.
    """
    if not STRATEGY_REGISTRY:
        print("Registry is empty. Registering 'SimpleMA' default strategy.")
        STRATEGY_REGISTRY["SimpleMA"] = simple_ma_strategy


def get_all_tickers() -> list[str]:
    """Flatten the TICKERS dict into a single list."""
    flat_tickers = []
    for _, tiks in TICKERS.items():
        flat_tickers.extend(tiks)
    return flat_tickers


def send_telegram(msg: str):
    """Send a message via Telegram Bot API."""
    if not TG_TOKEN:
        print(">>> SKIPPING TELEGRAM SEND (TG_TOKEN missing) <<<")
        print("--- Message Content ---")
        print(msg)
        print("-----------------------")
        return

    # Try to get Chat ID from env
    chat_id = os.getenv("TG_CHAT_ID")
    if not chat_id:
        print(">>> SKIPPING TELEGRAM SEND (TG_CHAT_ID missing) <<<")
        print("Please set TG_CHAT_ID in .env")
        print("--- Message Content ---")
        print(msg)
        print("-----------------------")
        return

    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}

    try:
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        print("✅ Telegram message sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")


# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------


def run_daily_job():
    print(f"=== Daily Runner Start ({date.today()}) ===")

    # 0. Setup
    ensure_strategies_registered()
    engine = BacktestEngine("quant.db")

    # 1. Update Data
    print("\n--- Step 1: Updating Market Data ---")
    tickers = get_all_tickers()
    sync_data(tickers)
    sync_fx_rates()

    # 2. Run Strategies & Generate Alerts
    print("\n--- Step 2: Calculating Signals ---")
    alerts = []

    for strategy_name, strategy_func in STRATEGY_REGISTRY.items():
        print(f"Strategy: {strategy_name}")

        for ticker in tickers:
            try:
                # Execute Strategy via Engine
                # Note: engine.run() returns a DF with [date, open, ..., signal, strategy_return, equity_curve]
                # It loads data from DB automatically.
                df_res = engine.run(ticker, strategy_func)

                if df_res.height < 2:
                    print(f"  [Skip] Not enough data for {ticker}")
                    continue

                # Extract last two rows (Yesterday -> Today)
                # rows = df_res.tail(2).to_dicts() # Polars to python dicts
                # Using tail(2) keeps columns.
                last_two = df_res.tail(2)

                # Check dates to ensure data is fresh?
                # User didn't strictly require date check, but it's good practice.
                # However, syncing might fail or market closed. We run on available data.

                today_row = last_two.row(1, named=True)
                prev_row = last_two.row(0, named=True)

                sig_today = today_row["signal"]
                sig_prev = prev_row["signal"]

                # Determine triggers
                # 1. Status Flip (Signal changed)
                # 2. Signal == Buy (1)

                is_flip = sig_today != sig_prev
                is_buy = sig_today == 1

                if is_flip or is_buy:
                    # Prepare Alert
                    symbol = today_row.get(
                        "ticker", ticker
                    )  # engine result might not preserve ticker column if strategy drops it, but engine loads it.
                    # Actually engine.run loads data then passes to strategy.
                    # If strategy returns DF without ticker col, engine doesn't re-add it?
                    # Engine.run returns: df_strategy + market_return + holding + strategy_return + equity_curve
                    # If simple_ma_strategy only appends columns, ticker col should be there.

                    price = today_row["close"]
                    date_str = str(today_row["date"])

                    # Formatting signal string
                    # If 'signal_str' exists, use it. Else map int.
                    sig_str = today_row.get(
                        "signal_str",
                        "BUY"
                        if sig_today == 1
                        else "SELL"
                        if sig_today == -1
                        else "HOLD",
                    )

                    reason_icons = []
                    if is_flip:
                        reason_icons.append("🔄Flip")
                    if is_buy:
                        reason_icons.append("🟢Buy")

                    msg = (
                        f"*{symbol}* ({strategy_name})\n"
                        f"Date: `{date_str}`\n"
                        f"Price: `{price:.2f}`\n"
                        f"Signal: **{sig_str}**\n"
                        f"Reasons: {' '.join(reason_icons)}"
                    )
                    alerts.append(msg)
                    print(f"  -> Alert Generated for {symbol}: {sig_str}")

            except Exception as e:
                print(f"  [Error] {ticker}: {e}")

    # 3. Send Summary
    print("\n--- Step 3: Dispatching Alerts ---")
    if alerts:
        header = f"🚀 **QuantBoard Daily Report** 🚀\n{date.today()}\n\n"
        full_body = header + "\n---\n".join(alerts)
        send_telegram(full_body)
    else:
        print("No actionable signals/alerts today.")

    print("\n=== Daily Runner Complete ===")


if __name__ == "__main__":
    run_daily_job()
