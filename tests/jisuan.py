import yfinance as yf
import akshare as ak
import polars as pl
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
from datetime import datetime
import matplotlib.pyplot as plt
import functools

# ==========================================
# 0. 环境设置
# ==========================================
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Arial Unicode MS",
    "PingFang SC",
    "Microsoft YaHei",
    "WenQuanYi Micro Hei",
]
plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# 1. 核心回测引擎 (Core Backtest Engine)
# ==========================================
def get_risk_contribution(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """计算风险贡献 (Risk Contribution)"""
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    if port_vol == 0:
        return np.zeros_like(w)
    mrc = np.dot(cov, w) / port_vol
    rc = w * mrc
    # Relative Risk Contribution
    rc_pct = rc / port_vol
    return rc_pct


def risk_parity_objective(w: np.ndarray, cov: np.ndarray) -> float:
    """风险平价优化目标函数"""
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    mrc = np.dot(cov, w) / port_vol
    rc = w * mrc
    rc_target = np.mean(rc)
    return np.sum(np.square(rc - rc_target)) * 1000


def run_threshold_strategy(
    returns_np: np.ndarray,
    dates_list: list,
    target_weights: np.ndarray,
    strategy_name: str = "Hybrid Rebalance",
    periodic_quarterly: bool = False,
    start_idx: int = 20,
) -> dict:
    """
    固定目标比例 + 混合阈值再平衡
    Trigger:
      1. Absolute Deviation > 5%
      2. Relative Deviation > 25%
      3. (Optional) Quarterly Rebalancing (Market Close of March, June, Sept, Dec)
    """
    T, N = returns_np.shape
    # window = 20 (Replaced by argument)

    if start_idx >= T:
        return {"dates": [], "values": [], "rebalance_count": 0}

    test_returns = returns_np[start_idx:]
    test_dates = dates_list[start_idx:]
    final_dates = [test_dates[0]]

    # Determine if weights are dynamic
    is_dynamic = False
    if target_weights.ndim == 2:
        if target_weights.shape[0] != len(test_returns):
            # Try to align? Or just raise error.
            # Assuming caller passed perfectly aligned weights matching `test_returns`.
            pass
        is_dynamic = True

    # Init Weights
    if is_dynamic:
        current_target = target_weights[0]
    else:
        current_target = target_weights

    weights = current_target.copy()

    portfolio_value = 1.0
    portfolio_values = [portfolio_value]
    rebalance_dates = []

    print(
        f"   [{strategy_name}] Start: {test_dates[0]} -> {test_dates[-1]} ({'Dynamic' if is_dynamic else 'Fixed'})"
    )

    for i in range(len(test_returns)):
        curr_date = test_dates[i]
        day_ret_vals = test_returns[i]

        # Update Target if dynamic
        if is_dynamic:
            current_target = target_weights[i]

        # 1. Update Portfolio Value
        # Portfolio Return = sum(w_i * r_i)
        # Handle Leverage & Financing Cost
        current_leverage = np.sum(weights)
        gross_ret = np.dot(weights, day_ret_vals)

        # Financing Cost: (Lev - 1) * CostRate
        # CostRate = 5.5% / 252
        daily_cost_rate = RISK_FREE_RATE_COST / 252
        cost = max(0, current_leverage - 1.0) * daily_cost_rate

        net_ret = gross_ret - cost
        portfolio_value *= 1 + net_ret
        portfolio_values.append(portfolio_value)
        final_dates.append(curr_date)

        # 2. Drift
        # w_i,t+1 = w_i,t * (1 + r_i,t) / (1 + NetPortRet)
        # Denominator uses Net Return to account for cost drag on NAV
        weights = weights * (1 + day_ret_vals) / (1 + net_ret)

        # 3. Check Thresholds
        abs_diff = np.abs(weights - current_target)
        cond_a = np.any(abs_diff > 0.05)

        rel_diff = np.abs(weights - current_target) / np.where(
            current_target < 1e-6, 1e-6, current_target
        )
        cond_b = np.any(rel_diff > 0.25)

        if cond_a or cond_b:
            weights = current_target.copy()
            rebalance_dates.append(curr_date)
            continue  # Rebalanced, skip periodic check

        # 4. Check Periodic (Quarterly)
        # Check if next day is in a new quarter, meaning today is the last trading day of the quarter (or effectively so)
        # Quarters end in: 3, 6, 9, 12.
        # Logic: If current month in [3,6,9,12] AND next date month != current month
        if periodic_quarterly:
            # Check if this is the last day of the list?
            if i < len(test_returns) - 1:
                next_date = test_dates[i + 1]
                # Assuming dates are datetime.date objects from polars cast
                curr_month = curr_date.month
                next_month = next_date.month

                if curr_month in [3, 6, 9, 12] and next_month != curr_month:
                    weights = current_target.copy()
                    rebalance_dates.append(curr_date)

    return {
        "dates": final_dates,
        "values": portfolio_values,
        "rebalance_count": len(rebalance_dates),
        "rebalance_dates": rebalance_dates,
    }


def run_periodic_strategy(
    returns_np: np.ndarray,
    dates_list: list,
    target_weights: np.ndarray,
    strategy_name: str = "Periodic Rebalance",
    period: str = "monthly",  # "monthly" or "quarterly"
    start_idx: int = 20,
) -> dict:
    """
    定期再平衡策略 (Periodic Rebalancing)
    Trigger:
      Only Rebalance at the end of Period (Month/Quarter)
    """
    T, N = returns_np.shape

    if start_idx >= T:
        return {"dates": [], "values": [], "rebalance_count": 0}

    test_returns = returns_np[start_idx:]
    test_dates = dates_list[start_idx:]
    final_dates = [test_dates[0]]

    # Determine if weights are dynamic
    is_dynamic = False
    if target_weights.ndim == 2:
        # Assuming caller passed perfectly aligned weights matching `test_returns`
        # However, check shape
        if target_weights.shape[0] != len(test_returns):
            # Fallback or warn?
            pass
        is_dynamic = True

    # Init Weights
    if is_dynamic:
        current_target = target_weights[0]
    else:
        current_target = target_weights

    weights = current_target.copy()

    portfolio_value = 1.0
    portfolio_values = [portfolio_value]
    rebalance_dates = []

    print(
        f"   [{strategy_name}] Start: {test_dates[0]} -> {test_dates[-1]} ({'Dynamic' if is_dynamic else 'Fixed'}, Period={period})"
    )

    for i in range(len(test_returns)):
        curr_date = test_dates[i]
        day_ret_vals = test_returns[i]

        # Update Target if dynamic
        if is_dynamic:
            current_target = target_weights[i]

        # 1. Update Portfolio Value
        current_leverage = np.sum(weights)
        gross_ret = np.dot(weights, day_ret_vals)

        # Financing Cost
        daily_cost_rate = RISK_FREE_RATE_COST / 252
        cost = max(0, current_leverage - 1.0) * daily_cost_rate

        net_ret = gross_ret - cost
        portfolio_value *= 1 + net_ret
        portfolio_values.append(portfolio_value)
        final_dates.append(curr_date)

        # 2. Drift
        weights = weights * (1 + day_ret_vals) / (1 + net_ret)

        # 3. Check Periodic Logic
        # logic: if next day is different month (or quarter), rebalance TODAY (conceptually)
        # We rebalance at close of this day to target weights
        should_rebalance = False
        if i < len(test_returns) - 1:
            next_date = test_dates[i + 1]
            curr_month = curr_date.month
            next_month = next_date.month

            if period == "monthly":
                if next_month != curr_month:
                    should_rebalance = True
            elif period == "quarterly":
                # End of Mar, Jun, Sep, Dec
                if curr_month in [3, 6, 9, 12] and next_month != curr_month:
                    should_rebalance = True

        if should_rebalance:
            weights = current_target.copy()
            rebalance_dates.append(curr_date)

    return {
        "dates": final_dates,
        "values": portfolio_values,
        "rebalance_count": len(rebalance_dates),
        "rebalance_dates": rebalance_dates,
    }


# Global Constants for Leverage
RISK_FREE_RATE_COST = 0.0556
MAX_LEVERAGE = 3.0
# Removed TARGET_VOL to avoid "assuming a volatility"
window = 252  # Global fallback


def run_strategy(
    returns_np: np.ndarray,
    dates_list: list,
    strategy_name: str = "Risk Parity",
) -> dict:
    """
    通用策略回测函数
    :param returns_np: T x N 收益率矩阵 (Simple Returns)
    :param dates_list: 日期列表
    :param strategy_name: 策略名称 ("Risk Parity" 或 "Equal Weight")
    :return: 结果字典 (Curve, Metrics, Weights info)
    """
    T, N = returns_np.shape
    window = 252

    # 初始化
    portfolio_value = 1.0
    portfolio_values = [portfolio_value]

    # 初始权重 (Unlevered)
    weights = np.ones(N) / N
    current_leverage = 1.0

    rebalance_dates = []

    # 模拟切片
    start_idx = window
    if start_idx >= T:
        print(f"   [Error] {strategy_name}: Data length {T} < window {window} + buffer")
        return {"dates": [], "values": [], "rebalance_count": 0}

    test_returns = returns_np[start_idx:]
    test_dates = dates_list[start_idx:]
    final_dates = [test_dates[0]]

    # 策略类型识别
    is_ew = "Equal Weight" in strategy_name

    # 融资成本 (每日)
    daily_cost_rate = RISK_FREE_RATE_COST / 252

    print(
        f"   [{strategy_name}] Start: {test_dates[0]} -> {test_dates[-1]} (Assets={N})"
    )

    for i in range(len(test_returns)):
        curr_date = test_dates[i]
        day_ret_vals = test_returns[i]

        # 1. Calculate PnL (Gross)
        # Gross Return = sum(w_i * r_i) * L
        gross_ret = np.dot(weights, day_ret_vals) * current_leverage

        # Financing Cost
        # Borrowed amount = (L - 1)
        # Cost = (L - 1) * daily_cost_rate
        cost = max(0, current_leverage - 1) * daily_cost_rate
        net_ret = gross_ret - cost

        portfolio_value *= 1 + net_ret

        portfolio_values.append(portfolio_value)
        final_dates.append(curr_date)

        # 2. Rebalance Logic
        # EW Logic: Reset to 1/N, Lev=1.0
        if is_ew:
            weights = np.ones(N) / N
            current_leverage = 1.0
            rebalance_dates.append(curr_date)
            continue

        # RP Logic: Optimization
        hist_slice = returns_np[start_idx + i - window + 1 : start_idx + i + 1]
        if len(hist_slice) < window:
            continue

        # 协方差 (Annualized)
        cov_matrix = np.cov(hist_slice, rowvar=False) * 252

        # A. Get Unlevered Weights (Sum=1, Mean VaRC)
        cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
        bnds = tuple((0.0, 1.0) for _ in range(N))

        res = minimize(
            risk_parity_objective,
            weights,  # Start guess
            args=(cov_matrix),
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            tol=1e-6,
        )

        if res.success:
            weights = res.x

            # B. Leverage Determination
            # User Requirement: Use leverage to ensure equal risk contribution,
            # NOT depending on highest volatility or assumed volatility.
            # Interpretation: Maximize the allocation (up to Max Limit) while maintaining parity.
            # i.e. Weights_Final = Weights_ERC * MAX_LEVERAGE
            current_leverage = MAX_LEVERAGE

            rebalance_dates.append(curr_date)

    return {
        "dates": final_dates,
        "values": portfolio_values,
        "rebalance_count": len(rebalance_dates),
        "rebalance_dates": rebalance_dates,
    }


def calculate_metrics(values: list) -> dict:
    """计算绩效指标 (Pure Python/Numpy)"""
    arr = np.array(values)
    total_ret = arr[-1] / arr[0] - 1

    # 假设每日数据
    days = len(arr)
    if days < 2:
        return {}

    cagr = (arr[-1] / arr[0]) ** (252 / days) - 1

    # Daily Returns
    pct_change = arr[1:] / arr[:-1] - 1
    vol = np.std(pct_change) * np.sqrt(252)

    sharpe = (cagr - 0.02) / vol if vol != 0 else 0

    # Max DD
    cum_max = np.maximum.accumulate(arr)
    dd = (arr - cum_max) / cum_max
    max_dd = np.min(dd)

    # --- PSR (Probabilistic Sharpe Ratio) ---
    # Reference: Bailey, D. H. and Lopez de Prado, M. (2012)
    # SR_est = mean / std (non-annualized)
    mu = np.mean(pct_change)
    sigma = np.std(pct_change)
    curr_skew = skew(pct_change) if sigma > 1e-6 else 0
    # Fisher=False returns Pearson kurtosis (normal=3.0)
    curr_kurt = kurtosis(pct_change, fisher=False) if sigma > 1e-6 else 3.0

    sr_std = mu / sigma if sigma > 1e-9 else 0

    # Target Sharpe (Benchmark) = 0
    sr_benchmark = 0

    # Denominator term
    # sigma_sr = sqrt( (1 - skew*SR + (kurt-1)/4 * SR^2) / (n-1) )
    n = len(pct_change)
    term1 = 1
    term2 = curr_skew * sr_std
    term3 = ((curr_kurt - 1) / 4) * (sr_std**2)
    numerator_var = term1 - term2 + term3

    if numerator_var < 0:
        # Should not happen theoretically unless data is weird
        numerator_var = 1.0

    sigma_sr = np.sqrt(numerator_var / (n - 1))

    if sigma_sr > 0:
        z_score = (sr_std - sr_benchmark) / sigma_sr
        psr = norm.cdf(z_score)
    else:
        psr = 0.0

    # --- Sortino Ratio ---
    # Downside deviation
    neg_rets = pct_change[pct_change < 0]
    downside_std = np.std(neg_rets) * np.sqrt(252) if len(neg_rets) > 0 else 0
    sortino = (cagr - 0.02) / downside_std if downside_std > 1e-6 else 0

    # --- Omega Ratio ---
    # Threshold = 0
    # Omega = sum(returns > 0) / abs(sum(returns < 0))
    threshold = 0
    gains = pct_change[pct_change > threshold]
    losses = pct_change[pct_change < threshold]
    sum_gains = np.sum(gains)
    sum_losses = np.sum(losses)
    omega = sum_gains / abs(sum_losses) if abs(sum_losses) > 1e-9 else float("inf")

    # --- Calmar Ratio ---
    # CAGR / MaxDD
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0

    # --- Ulcer Index ---
    # Sqrt(mean(drawdowns^2))
    # Drawdowns are percentages e.g. 0.05 for 5%
    # dd array calculated above: (arr - cum_max) / cum_max
    # We square them, mean, sqrt
    ulcer_index = np.sqrt(np.mean(dd**2))

    return {
        "Total Return": f"{total_ret:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Volatility": f"{vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "MaxDD": f"{max_dd:.2%}",
        "Sortino": f"{sortino:.2f}",
        "Omega": f"{omega:.2f}",
        "Calmar": f"{calmar:.2f}",
        "Ulcer": f"{ulcer_index:.4f}",
        "PSR": f"{psr:.2%}",
    }


# ==========================================
# 2. 数据获取与处理
# ==========================================
print(">>> [Step 1] 启动混合数据引擎...")

start_date_str = "20200301"
end_date_str = datetime.now().strftime("%Y%m%d")

# 资产定义
# US: FLIN, QQQ, GLD, IEF, DBMF
usd_assets = {
    "FLIN": "FLIN",
    "QQQ": "QQQ",
    "GLD": "GLD",
    "IEF": "IEF",
    "DBMF": "DBMF",
    "SPY": "SPY",
}
# CN:
cn_assets = {"161115": "161115", "515450": "515450", "511380": "511380"}
fx_ticker = "CNY=X"

# --- A. 获取 CN ---
cn_dfs = []
for code, name in cn_assets.items():
    try:
        df_pd = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date=start_date_str,
            end_date=end_date_str,
            adjust="qfq",
        )
        # 用 Polars 接管
        # Need High/Low for Parkinson Volatility
        df_pl = pl.from_pandas(df_pd).select(
            [
                pl.col("日期").cast(pl.Date).alias("Date"),
                pl.col("收盘").alias(code),
                pl.col("最高").alias(f"{code}_HIGH"),
                pl.col("最低").alias(f"{code}_LOW"),
            ]
        )
        cn_dfs.append(df_pl)
    except Exception as e:
        print(f"   [Error] {code}: {e}")

if len(cn_dfs) > 1:
    df_cn = functools.reduce(
        lambda a, b: a.join(b, on="Date", how="full", coalesce=True), cn_dfs
    )
else:
    if len(cn_dfs) > 0:
        df_cn = cn_dfs[0]
    else:
        print("[Error] No CN assets fetched. Exiting.")
        exit(1)

# --- B. 获取 US ---
yf_tickers = list(usd_assets.values()) + [fx_ticker]
yf_start = f"{start_date_str[:4]}-{start_date_str[4:6]}-{start_date_str[6:]}"

# yf usually returns pandas
df_us_pd = yf.download(yf_tickers, start=yf_start, auto_adjust=True, progress=False)

# 降维处理 MultiIndex
# 降维处理 MultiIndex
if isinstance(df_us_pd.columns, tuple) or (
    hasattr(df_us_pd.columns, "nlevels") and df_us_pd.columns.nlevels > 1
):
    # Flatten MultiIndex if present
    # Expecting: (Price, Ticker) -> We need to be careful
    # Recent yfinance often returns MultiIndex: columns = [(Price, Ticker), ...]
    pass


# Reset index to get Date column
df_us_pd = df_us_pd.reset_index()

# Flatten columns manually to ensure cleaner usage
# e.g. Close -> QQQ, High -> QQQ_HIGH
new_cols = []
cols = df_us_pd.columns
for c in cols:
    if isinstance(c, tuple):
        # (PriceType, Ticker)
        price_type, ticker = c
        if price_type == "Close":
            new_cols.append(ticker)
        elif price_type == "High":
            new_cols.append(f"{ticker}_HIGH")
        elif price_type == "Low":
            new_cols.append(f"{ticker}_LOW")
        elif (
            price_type == "Date" or ticker == ""
        ):  # Handle Index name usually '' or 'Date'
            new_cols.append("Date")
        else:
            # Ignore Volume etc for now unless matched
            new_cols.append(f"{ticker}_{price_type}")
    else:
        # Flat index already
        new_cols.append(c)

df_us_pd.columns = new_cols

# Polars conversion (Ensure we select only what we need to avoid dupes)
# We need Ticker, Ticker_HIGH, Ticker_LOW for each US asset
sel_cols = ["Date"]
for t in yf_tickers:
    if t in df_us_pd.columns:
        sel_cols.append(t)
    if f"{t}_HIGH" in df_us_pd.columns:
        sel_cols.append(f"{t}_HIGH")
    if f"{t}_LOW" in df_us_pd.columns:
        sel_cols.append(f"{t}_LOW")

df_us = pl.from_pandas(df_us_pd[sel_cols]).with_columns(pl.col("Date").dt.date())

# --- C. Merge & Clean ---
df_merged = df_cn.join(df_us, on="Date", how="full", coalesce=True).sort("Date")
print(">>> [Debug] Merged Data Null Counts:")
print(df_merged.null_count())
print(
    f">>> [Debug] Data Time Range: {df_merged['Date'].min()} -> {df_merged['Date'].max()}"
)

# Handle Holidays: Forward Fill (Use previous close if market is closed)
# This prevents losing days where one market is open and the other is closed (e.g. CNY vs Christmas)
df_all = df_merged.fill_null(strategy="forward").drop_nulls()

print(f">>> [Step 2] Cleaned Data Rows: {df_all.height}")
if df_all.height == 0:
    print(
        ">>> [Critical Error] Intersection is EMPTY! Check if Assets overlap in time."
    )
    exit(1)

# ==========================================
# 3. 因子计算
# ==========================================

asset_cols_final = []  # 最终要用的 columns

# 3.1 汇率处理
if fx_ticker not in df_all.columns:
    raise ValueError("Missing FX data")

exprs = [pl.col("Date")]

# CN 资产直接保留
for code in cn_assets:
    exprs.append(pl.col(code))
    asset_cols_final.append(code)

# US 资产乘汇率
for name, ticker in usd_assets.items():
    # 生成如 'QQQ' 列
    exprs.append((pl.col(ticker) * pl.col(fx_ticker)).alias(name))
    asset_cols_final.append(name)

    # 同时也处理 High/Low (用于 Volatility 计算)
    # FX adjusted: High_USD * FX, Low_USD * FX
    # 注意: 这样做只是近似（FX波动在日内被忽略），但作为波动率估计已足够
    # 如果源数据有 HIGH/LOW
    if f"{ticker}_HIGH" in df_all.columns and f"{ticker}_LOW" in df_all.columns:
        exprs.append(
            (pl.col(f"{ticker}_HIGH") * pl.col(fx_ticker)).alias(f"{name}_HIGH")
        )
        exprs.append((pl.col(f"{ticker}_LOW") * pl.col(fx_ticker)).alias(f"{name}_LOW"))

# CN 资产 High/Low 也要保留
for code in cn_assets:
    if f"{code}_HIGH" in df_all.columns:
        exprs.append(pl.col(f"{code}_HIGH"))
    if f"{code}_LOW" in df_all.columns:
        exprs.append(pl.col(f"{code}_LOW"))

df_prices = df_all.select(exprs)

# 3.2 计算简单收益率 (Simple Returns)
# r = P_t / P_{t-1} - 1
ret_exprs = [(pl.col(c) / pl.col(c).shift(1) - 1).alias(c) for c in asset_cols_final]
df_returns = df_prices.select([pl.col("Date")] + ret_exprs).drop_nulls()

print(f">>> [Debug] df_returns height: {df_returns.height}")
if df_returns.height < 255:
    print("Warning: Data insufficient for 252 day window + initial steps.")


# ==========================================
# 4. 运行四种情景
# ==========================================
# 定义资产池
base_assets = list(cn_assets.keys()) + [
    "QQQ",
    "GLD",
    "IEF",
    "FLIN",
]
full_assets = base_assets + ["DBMF"]

# 准备 Numpy 数据
dates_full = df_returns["Date"].to_list()
results = {}

# --- Run 1: Equal Weight (Base) ---
# --- Run 1: Equal Weight (Base) ---
cols = base_assets
data = df_returns.select(cols).to_numpy()
n_base = len(base_assets)
target_base = np.ones(n_base) / n_base
results["Equal Weight (Base)"] = run_threshold_strategy(
    data, dates_full, target_base, "Equal Weight (Base)"
)

# --- Run 2: Equal Weight (Full) ---
cols = full_assets
data = df_returns.select(cols).to_numpy()
n_full = len(full_assets)
target_full = np.ones(n_full) / n_full
results["Equal Weight (Full)"] = run_threshold_strategy(
    data, dates_full, target_full, "Equal Weight (Full)"
)

# # --- Run 3: RP (Base) ---
# cols = base_assets
# data = df_returns.select(cols).to_numpy()
# results["RP (Base)"] = run_strategy(data, dates_full, "RP (Base)")
#
# # --- Run 4: RP (Full) ---
# cols = full_assets
# data = df_returns.select(cols).to_numpy()
# results["RP (Full)"] = run_strategy(data, dates_full, "RP (Full)")

# --- Run 5: Hybrid Threshold (Fixed) ---
# Target Weights:
# FLIN=14.33%, 515450=14.19%, QQQ=16.09%, GLD=9.82%,
# IEF=4.24%, 161115=18.10%, 511380=23.22%
target_map = {
    "FLIN": 0.1433,
    "515450": 0.1419,
    "QQQ": 0.1609,
    "GLD": 0.0982,
    "IEF": 0.0424,
    "161115": 0.1810,
    "511380": 0.2322,
}
# Only use these assets
hybrid_assets = list(target_map.keys())
# Ensure column order matches the target vector order
data_hybrid = df_returns.select(hybrid_assets).to_numpy()
target_vec = np.array([target_map[c] for c in hybrid_assets])
# Normalize to exactly 1.0 just in case
target_vec = target_vec / np.sum(target_vec)

results["Hybrid (Fixed)"] = run_threshold_strategy(
    data_hybrid, dates_full, target_vec, "Hybrid (Fixed)"
)

# --- Run 5b: Hybrid (DBMF 8.1%) ---
# Explicitly defined weights (splitting 161115's 18.1% -> 10% + 8.1% DBMF)
target_map_dbmf = {
    "FLIN": 0.1433,
    "515450": 0.1419,
    "QQQ": 0.1609,
    "GLD": 0.0982,
    "IEF": 0.0424,
    "161115": 0.1000,
    "511380": 0.2322,
    "DBMF": 0.0810,
}
hybrid_dbmf_assets = list(target_map_dbmf.keys())
data_hybrid_dbmf = df_returns.select(hybrid_dbmf_assets).to_numpy()
target_vec_dbmf = np.array([target_map_dbmf[c] for c in hybrid_dbmf_assets])
# Re-normalize slightly if needed (sum is ~0.9999)
target_vec_dbmf = target_vec_dbmf / np.sum(target_vec_dbmf)

results["Hybrid (DBMF 8.1%)"] = run_threshold_strategy(
    data_hybrid_dbmf, dates_full, target_vec_dbmf, "Hybrid (DBMF 8.1%)"
)

# --- Run 5c: Hybrid (DBMF 13.1%) ---
# 161115=5%, DBMF=13.1%
target_map_dbmf_v2 = {
    "FLIN": 0.1433,
    "515450": 0.1419,
    "QQQ": 0.1609,
    "GLD": 0.0982,
    "IEF": 0.0424,
    "161115": 0.0500,
    "511380": 0.2322,
    "DBMF": 0.1310,
}
# Use same asset list logic as above, just different map
hybrid_dbmf_v2_assets = list(target_map_dbmf_v2.keys())
data_hybrid_dbmf_v2 = df_returns.select(hybrid_dbmf_v2_assets).to_numpy()
target_vec_dbmf_v2 = np.array([target_map_dbmf_v2[c] for c in hybrid_dbmf_v2_assets])
# Normalize to exactly 1.0
target_vec_dbmf_v2 = target_vec_dbmf_v2 / np.sum(target_vec_dbmf_v2)

results["Hybrid (DBMF 13.1%)"] = run_threshold_strategy(
    data_hybrid_dbmf_v2, dates_full, target_vec_dbmf_v2, "Hybrid (DBMF 13.1%)"
)

# --- Run 6: Inverse Volatility (Parkinson EWMA) ---
# 1. Calculate Parkinson Variance for each hybrid asset
# V = (1 / 4ln2) * (ln(H/L))^2
k = 1.0 / (4.0 * np.log(2.0))
inv_vol_assets = hybrid_assets  # FLIN, 515450, QQQ, GLD, IEF, 161115, 511380

parkinson_exprs = []
for asset in inv_vol_assets:
    # High/Low columns need to be present
    h_col = f"{asset}_HIGH"
    l_col = f"{asset}_LOW"

    # Parkinson Variance
    variance_expr = ((pl.col(h_col) / pl.col(l_col)).log().pow(2) * k).alias(
        f"{asset}_VAR"
    )

    parkinson_exprs.append(variance_expr)

# Calculate Daily Variances
df_vars = df_prices.select([pl.col("Date")] + parkinson_exprs)

# 2. EWMA Smoothing per asset
# Use a shorter window for Volatility calculation (Decoupled from Global Backtest Window)
vol_window = 60
lambda_param = 0.94
alpha = 1 - lambda_param

ewma_exprs = []
for asset in inv_vol_assets:
    # EWM on Variance
    # We use min_samples=vol_window (60) to get values faster
    ewma_exprs.append(
        pl.col(f"{asset}_VAR")
        .ewm_mean(alpha=alpha, adjust=False, min_samples=vol_window)
        .alias(f"{asset}_EWMA_VAR")
    )

df_ewma = df_vars.select(ewma_exprs)
# Convert to Numpy for weight calc (Shape: T x N)
ewma_vars = df_ewma.to_numpy()

# 3. Calculate Weights
# Vol = sqrt(EWMA_VAR)
# Weight = (1/Vol) / Sum(1/Vol)
vols = np.sqrt(ewma_vars)
# Avoid div by zero / NaNs
vols[vols == 0] = np.inf
vols[np.isnan(vols)] = np.inf

inv_vols = 1.0 / vols

# --- Penalty for Bond Assets (IEF, 161115) ---
# Multiply inverse volatility by 0.1 to suppress allocation
penalty_assets = ["IEF", "161115"]
for pa in penalty_assets:
    if pa in inv_vol_assets:
        idx = inv_vol_assets.index(pa)
        # Apply penalty: reduce inv_vol => reduce weight
        inv_vols[:, idx] = inv_vols[:, idx] * 0.1

# Normalize row-wise
sum_inv_vols = np.sum(inv_vols, axis=1, keepdims=True)
# Handle rows where sum is 0 or NaN (e.g. initial warm-up period)
sum_inv_vols[sum_inv_vols == 0] = 1.0
sum_inv_vols[np.isnan(sum_inv_vols)] = 1.0

dynamic_weights = inv_vols / sum_inv_vols

# 4. Padding / Alignment
# dynamic_weights might have 0s or NaNs in the first 'vol_window' rows.
# Global backtest starts at 'window' (252).
# We ensure the ENTIRE array is valid so slicing works perfectly.
# Fill initial invalid weights with Equal Weight
n_assets = len(inv_vol_assets)
equal_w = np.ones(n_assets) / n_assets

# Identify invalid rows (sum approx 0)
invalid_mask = np.sum(dynamic_weights, axis=1) == 0
dynamic_weights[invalid_mask] = equal_w

# Also check for NaNs just in case
dynamic_weights = np.nan_to_num(dynamic_weights, nan=1.0 / n_assets)

# Renormalize to ensure sum=1.0
row_sums = np.sum(dynamic_weights, axis=1, keepdims=True)
dynamic_weights = dynamic_weights / row_sums

# 5. Pass FULL length arrays to run_threshold_strategy
# The function will slice both returns and weights starting from `start_idx`.
# We ensure dynamic_weights matches T (len of df_returns).
# Since df_ewma came from df_vars which came from df_prices (same length as df_returns), it is aligned.

start_idx_backtest = 60  # Keep Global Window for fair comparison
# We need to pass the weights that align with the TEST period if we were passing sliced weights.
# BUT, looking at run_threshold_strategy:
#   test_returns = returns_np[start_idx:]
#   target_weights[0] refers to the first day of TEST period?
#   Let's check logic:
#   if target_weights.ndim == 2:
#       current_target = target_weights[i]
#
#   Wait, 'run_threshold_strategy' iterates i in range(len(test_returns)).
#   If we pass a 2D array, it expects shape (Len_Test, N).
#   So we MUST slice dynamic_weights here to match [start_idx:].

target_weights_aligned = dynamic_weights[start_idx_backtest:]

results["InvVol (Parkinson)"] = run_periodic_strategy(
    data_hybrid,
    dates_full,
    target_weights_aligned,
    "InvVol (Parkinson)",
    period="monthly",
    start_idx=start_idx_backtest,
)

# # --- Run 7: InvVol (Parkinson) 3x Dynamic ---
# # Target Vol = 15%
# # L_t = min(3.0, 0.15 / Sigma_Port)
# TARGET_VOL = 0.15
# LEV_CAP = 3.0
#
# # We need to calculate Sigma_Port for every day in the test period
# # using the trailing correlation matrix and the InvVol weights.
# # The `run_threshold_strategy` starts from `start_idx`.
# # We need to compute determining factors aligned with that.
# # `target_weights_aligned` already matches the period [start_idx:].
# # `data_hybrid` is the full dataset (T, N).
# # We need returns for covariance calculation.
#
# dynamic_lev_weights = []
# inv_vol_base_weights = target_weights_aligned  # Base 1.0 sum weights
#
# # Calculate Cov and Vol iteratively
# # Need to align indices carefully.
# # test_dates start at index `window` (252).
# # `target_weights_aligned` starts at index 0 (relative to test start).
#
# print(">>> [Step 3] Calculating Dynamic Leverage for InvVol...")
# for i in range(len(target_weights_aligned)):
#     # Current time index in full arrays
#     current_full_idx = window + i
#
#     # Base weight for this day
#     base_w = inv_vol_base_weights[i]
#
#     # Get history for Covariance
#     # Slice: [current - window : current]
#     hist_slice = data_hybrid[current_full_idx - window : current_full_idx]
#
#     if len(hist_slice) < 20:  # Safety
#         lev = 1.0
#     else:
#         # Annualized Covariance
#         cov = np.cov(hist_slice, rowvar=False) * 252
#
#         # Portfolio Vol = sqrt(w' Sigma w)
#         port_var = np.dot(base_w.T, np.dot(cov, base_w))
#         port_vol = np.sqrt(port_var)
#
#         # Required Leverage
#         if port_vol == 0:
#             lev = LEV_CAP
#         else:
#             lev = TARGET_VOL / port_vol
#
#         # Cap
#         lev = min(LEV_CAP, lev)
#
#     # Apply Leverage
#     final_w = base_w * lev
#     dynamic_lev_weights.append(final_w)
#
# dynamic_lev_weights = np.array(dynamic_lev_weights)
#
# results["InvVol (3x Dynamic)"] = run_threshold_strategy(
#     data_hybrid, dates_full, dynamic_lev_weights, "InvVol (3x Dynamic)"
# )

# # --- Run 8: User Hybrid (3x Leveraged) ---
# # Weights: FLIN=14.33%, 515450=14.19%, QQQ=16.09%, GLD=9.82%, IEF=42.4%, 161115=181%, 511380=23.22%
# user_target_map = {
#     "FLIN": 0.1433,
#     "515450": 0.1419,
#     "QQQ": 0.1609,
#     "GLD": 0.0982,
#     "IEF": 0.424,
#     "161115": 1.81,
#     "511380": 0.2322,
# }
# user_hybrid_assets = list(user_target_map.keys())
# data_user_hybrid = df_returns.select(user_hybrid_assets).to_numpy()
# user_target_vec = np.array([user_target_map[c] for c in user_hybrid_assets])
# # Note: No Normalization! Sum is ~3.01
#
# results["User Hybrid (3x)"] = run_threshold_strategy(
#     data_user_hybrid, dates_full, user_target_vec, "User Hybrid (3x)"
# )

# --- Run 9: Traditional 60/40 ---
# Assets: SPY (60%), IEF (40%)
strat_6040_map = {"SPY": 0.60, "IEF": 0.20, "161115": 0.20}
assets_6040 = list(strat_6040_map.keys())
data_6040 = df_returns.select(assets_6040).to_numpy()
target_6040 = np.array([strat_6040_map[c] for c in assets_6040])

results["Traditional 60/40"] = run_threshold_strategy(
    data_6040, dates_full, target_6040, "Traditional 60/40"
)

# --- Run 10: 60/20/20 (Nasdaq/Gold/Bond) ---
# Assets: QQQ (60%), GLD (20%), IEF (20%)
strat_602020_map = {"QQQ": 0.60, "GLD": 0.20, "IEF": 0.10, "161115": 0.10}
assets_602020 = list(strat_602020_map.keys())
data_602020 = df_returns.select(assets_602020).to_numpy()
target_602020 = np.array([strat_602020_map[c] for c in assets_602020])

results["60/20/20 (N/G/B)"] = run_threshold_strategy(
    data_602020, dates_full, target_602020, "60/20/20 (N/G/B)"
)

# --- Run 11: Equal Weight (Quarterly + Threshold) ---
# Same assets as Equal Weight (Full)
cols = full_assets
data = df_returns.select(cols).to_numpy()
n_full = len(full_assets)
target_full = np.ones(n_full) / n_full

results["EW (Quarterly + Threshold)"] = run_threshold_strategy(
    data,
    dates_full,
    target_full,
    "EW (Quarterly + Threshold)",
    periodic_quarterly=True,
)


# --- Run 11: Traditional 80/20 ---
# Assets: SPY (80%), IEF (20%)
strat_8020_map = {"SPY": 0.80, "IEF": 0.10, "161115": 0.10}
assets_8020 = list(strat_8020_map.keys())
data_8020 = df_returns.select(assets_8020).to_numpy()
target_8020 = np.array([strat_8020_map[c] for c in assets_8020])

results["Traditional 80/20"] = run_threshold_strategy(
    data_8020, dates_full, target_8020, "Traditional 80/20"
)


# ==========================================
# 5. 结果展示
# ==========================================
print("\n" + "=" * 70)
print(f">>> Active Strategies: {list(results.keys())}")
print(
    f"{'Strategy':<25} | {'TotalRet':<9} | {'CAGR':<7} | {'Sharpe':<6} | {'Sortino':<7} | {'Omega':<6} | {'Calmar':<6} | {'MaxDD':<8} | {'Ulcer':<6} | {'PSR':<7} | {'Rebal':<5}"
)
print("-" * 125)

for name, res in results.items():
    m = calculate_metrics(res["values"])
    rebal = res["rebalance_count"]
    print(
        f"{name:<25} | {m['Total Return']:<9} | {m['CAGR']:<7} | {m['Sharpe']:<6} | {m['Sortino']:<7} | {m['Omega']:<6} | {m['Calmar']:<6} | {m['MaxDD']:<8} | {m['Ulcer']:<6} | {m['PSR']:<7} | {rebal:<5}"
    )

print("=" * 70)

# ==========================================
# 6. 图表绘制
# ==========================================
plt.figure(figsize=(12, 6))

colors = {
    "Equal Weight (Base)": "gray",
    "Equal Weight (Full)": "black",
    "RP (Base)": "#1f77b4",  # Blue
    "RP (Full)": "#d62728",  # Red
    "Hybrid (Fixed)": "#2ca02c",  # Green
    "Hybrid (DBMF 8.1%)": "#8c564b",  # Brown
    "InvVol (Parkinson)": "#9467bd",  # Purple
    "InvVol (3x Dynamic)": "#ff7f0e",  # Orange
    "User Hybrid (3x)": "#e377c2",  # Pink
    "Traditional 60/40": "tab:cyan",
    "60/20/20 (N/G/B)": "tab:olive",
    "Traditional 80/20": "tab:brown",
    "EW (Quarterly + Threshold)": "tab:blue",
}
styles = {
    "Equal Weight (Base)": "--",
    "Equal Weight (Full)": "-",
    "RP (Base)": "--",
    "RP (Full)": "-",
    "Hybrid (Fixed)": "-",
    "Hybrid (DBMF 8.1%)": "-",
    "InvVol (Parkinson)": "--",
    "InvVol (3x Dynamic)": "-",
    "User Hybrid (3x)": "-",
    "Traditional 60/40": "-",
    "60/20/20 (N/G/B)": "-",
    "Traditional 80/20": "-",
    "EW (Quarterly + Threshold)": "--",
}

for name, res in results.items():
    # Dates usually start from index 1 relative to original values list?
    # run_strategy logic: returns 'values' list which matches 'dates' list length (both built in loop + initial)
    # The 'final_dates' has T+1 items (initial + loop). 'portfolio_values' has T+1 items.
    # We should verify length.

    x_dates = res["dates"]
    y_vals = res["values"]
    rebal_dates = res.get("rebalance_dates", [])

    # Plot Curve
    plt.plot(
        x_dates,
        y_vals,
        label=name,
        color=colors.get(name, "black"),
        linestyle=styles.get(name, "-"),
        linewidth=2 if "(Full)" in name else 1.5,
    )

    # Plot Rebalancing Markers
    # Filter out daily rebalancing strategies to avoid clutter (e.g. if rebalance > 99% of days)
    if len(rebal_dates) > 0 and (len(rebal_dates) / len(x_dates) < 0.99):
        np_dates = np.array(x_dates)
        np_vals = np.array(y_vals)

        rebal_set = set(rebal_dates)
        rebal_indices = [i for i, d in enumerate(x_dates) if d in rebal_set]

        if rebal_indices:
            rx = np_dates[rebal_indices]
            ry = np_vals[rebal_indices]

            plt.scatter(
                rx,
                ry,
                s=60,
                marker="^",
                edgecolors="black",
                facecolors=colors.get(name, "black"),
                zorder=10,
            )

plt.title("Risk Parity & Equal Weight: DBMF Impact Analysis (Daily Rebalancing)")
plt.xlabel("Date")
plt.ylabel("Portfolio Equity (Normalized)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()
