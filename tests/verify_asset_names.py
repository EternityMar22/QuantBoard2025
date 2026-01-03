import akshare as ak
import yfinance as yf
import pandas as pd


def check_cn_assets(codes):
    """
    Check CN ETF assets using Akshare.
    """
    print(f"Checking CN Assets: {codes}")
    try:
        # Get all ETF spot data
        df = ak.fund_etf_spot_em()
        # Filter
        results = {}
        for code in codes:
            row = df[df["代码"] == code]
            if not row.empty:
                name = row["名称"].values[0]
                results[code] = name
            else:
                results[code] = "Not Found in EM Spot Data"

        return results
    except Exception as e:
        return {c: f"Error: {e}" for c in codes}


def search_cn_fallback(code):
    """Fallback search in LOF or other lists"""
    try:
        # Try LOF spot
        lof_df = ak.fund_lof_spot_em()
        row = lof_df[lof_df["代码"] == code]
        if not row.empty:
            return row["名称"].values[0]

        # Try Bond ETF or just fund info (slow but specific)
        # ak.fund_name_em() is too big.
        # Try fund_individual_basic_info for a guess (might fail if not open)

        return "Not Found"
    except:
        return "Error in Fallback"


def check_us_assets(tickers):
    """
    Check US assets using yfinance.
    """
    print(f"Checking US Assets: {tickers}")
    results = {}
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            # Try to get longName or shortName
            info = ticker.info
            name = info.get("longName") or info.get("shortName") or "Name Not Found"
            results[t] = name
        except Exception as e:
            results[t] = f"Error: {e}"
    return results


if __name__ == "__main__":
    cn_codes = ["161115", "515450", "511380"]
    us_tickers = ["FLIN", "QQQ", "GLD", "IEF"]

    print("--- Verifying Asset Names ---")

    cn_res = check_cn_assets(cn_codes)
    for k, v in cn_res.items():
        if "Not Found" in v or "Error" in v:
            fallback = search_cn_fallback(k)
            if fallback != "Not Found" and "Error" not in fallback:
                cn_res[k] = fallback + " (Found in Fallback/LOF)"
        print(f"[CN] {k}: {cn_res[k]}")

    print("-" * 20)

    us_res = check_us_assets(us_tickers)
    for k, v in us_res.items():
        print(f"[US] {k}: {v}")

    print("--- Done ---")
