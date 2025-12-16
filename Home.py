"""
QuantBoard 2025 - 信号仪表盘
============================
现代化金融信号监控界面，支持多市场分类视图
"""

import streamlit as st
import duckdb
import polars as pl
from datetime import datetime
from src.config import TICKERS
from src.strategies.demo_strategy import simple_ma_strategy

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantBoard 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Database Helper
# -----------------------------------------------------------------------------
def get_db_connection():
    """建立 DuckDB 只读连接"""
    return duckdb.connect("quant.db", read_only=True)


@st.cache_data(ttl=300)  # 缓存 5 分钟
def get_last_update_time() -> str:
    """获取数据库中最新的数据日期"""
    try:
        with get_db_connection() as con:
            tables = con.sql("SHOW TABLES").pl()
            if "market_data" not in tables["name"].to_list():
                return "无数据"
            result = con.sql("SELECT MAX(date) FROM market_data").fetchone()
            if result and result[0]:
                return str(result[0])
            return "无数据"
    except Exception as e:
        return f"错误: {e}"


@st.cache_data(ttl=60)  # 缓存 1 分钟
def load_ticker_data(ticker: str) -> pl.DataFrame:
    """读取指定 Ticker 的最近数据 (用于计算信号)"""
    query = f"""
        SELECT date, close 
        FROM market_data 
        WHERE ticker = '{ticker}' 
        ORDER BY date DESC 
        LIMIT 60
    """
    try:
        with get_db_connection() as con:
            df = con.sql(query).pl()
        if df.is_empty():
            return pl.DataFrame()
        return df.sort("date")
    except Exception:
        return pl.DataFrame()


def get_sparkline_data(ticker: str) -> list[float]:
    """获取迷你图数据 (最近 20 个收盘价)"""
    query = f"""
        SELECT close 
        FROM market_data 
        WHERE ticker = '{ticker}' 
        ORDER BY date DESC 
        LIMIT 20
    """
    try:
        with get_db_connection() as con:
            df = con.sql(query).pl()
        if df.is_empty():
            return []
        # 反转为时间正序
        return df["close"].reverse().to_list()
    except Exception:
        return []


def compute_signal(ticker: str, market: str) -> dict:
    """计算单个标的的信号与指标"""
    df_data = load_ticker_data(ticker)

    if df_data.height < 20:
        return {
            "市场": market,
            "标的代码": ticker,
            "最新价格": None,
            "今日信号": "📊 数据不足",
            "信号值": 0,  # 用于统计
            "20日涨跌幅": None,
            "价格走势": [],
        }

    # 运行策略
    df_processed = simple_ma_strategy(df_data)
    latest_row = df_processed.tail(1)
    current_price = latest_row["close"][0]
    signal_raw = latest_row["signal_str"][0]

    # 信号映射 (带 Emoji)
    signal_map = {
        "买入": ("🟢 买入", 1),
        "卖出": ("🔴 卖出", -1),
        "持有": ("⚪ 持有", 0),
    }
    signal_display, signal_val = signal_map.get(signal_raw, (signal_raw, 0))

    # 计算 20 日涨跌幅
    pct_chg_20d = 0.0
    if df_processed.height >= 21:
        price_20d_ago = df_processed["close"][-21]
        if price_20d_ago != 0:
            pct_chg_20d = (current_price - price_20d_ago) / price_20d_ago

    # 获取迷你图数据
    sparkline = get_sparkline_data(ticker)

    return {
        "市场": market,
        "标的代码": ticker,
        "最新价格": current_price,
        "今日信号": signal_display,
        "信号值": signal_val,
        "20日涨跌幅": pct_chg_20d,
        "价格走势": sparkline,
    }


def display_signal_table(data: list[dict]) -> None:
    """展示信号数据表格"""
    if not data:
        st.info("暂无数据")
        return

    df = pl.DataFrame(data)

    st.dataframe(
        df.select(
            ["市场", "标的代码", "最新价格", "今日信号", "20日涨跌幅", "价格走势"]
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            "市场": st.column_config.TextColumn("市场", width="small"),
            "标的代码": st.column_config.TextColumn("标的代码", width="small"),
            "最新价格": st.column_config.NumberColumn(
                "最新价格",
                format="%.2f",
            ),
            "今日信号": st.column_config.TextColumn("今日信号", width="medium"),
            "20日涨跌幅": st.column_config.ProgressColumn(
                "20日涨跌幅",
                format="%.1f%%",
                min_value=-0.5,
                max_value=0.5,
            ),
            "价格走势": st.column_config.LineChartColumn(
                "20日走势",
                width="medium",
            ),
        },
    )


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("📈 QuantBoard 2025")
st.caption(
    f"个人量化信号仪表盘 · 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 系统状态")
    last_update = get_last_update_time()
    st.info(f"📅 数据更新日期\n\n**{last_update}**")

    # 统计信息将在主逻辑后填充
    stats_placeholder = st.empty()

    st.divider()

    st.header("🚀 快捷入口")
    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.page_link("pages/01_Portfolio.py", label="⚖️ 持仓再平衡", icon="📊")


# -----------------------------------------------------------------------------
# Main Logic - 计算所有信号
# -----------------------------------------------------------------------------
results: list[dict] = []
market_names = {"US": "🇺🇸 美股", "CN": "🇨🇳 A股"}

# 进度指示
all_tickers = [
    (market, ticker) for market, tickers in TICKERS.items() for ticker in tickers
]
progress_bar = st.progress(0, text="正在加载信号数据...")

for i, (market, ticker) in enumerate(all_tickers):
    progress_bar.progress((i + 1) / len(all_tickers), text=f"分析 {ticker}...")
    result = compute_signal(ticker, market_names.get(market, market))
    results.append(result)

progress_bar.empty()


# -----------------------------------------------------------------------------
# 统计指标
# -----------------------------------------------------------------------------
buy_count = sum(1 for r in results if r["信号值"] == 1)
sell_count = sum(1 for r in results if r["信号值"] == -1)
hold_count = sum(1 for r in results if r["信号值"] == 0 and r["最新价格"] is not None)
total_count = len(results)

# 更新侧边栏统计
with stats_placeholder.container():
    st.metric("📊 监控标的", total_count)


# -----------------------------------------------------------------------------
# 指标卡片区
# -----------------------------------------------------------------------------
st.subheader("📊 信号概览")

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "🟢 买入信号",
    buy_count,
    delta=f"{buy_count}" if buy_count > 0 else None,
    delta_color="normal",
)
col2.metric(
    "🔴 卖出信号",
    sell_count,
    delta=f"{sell_count}" if sell_count > 0 else None,
    delta_color="inverse",
)
col3.metric("⚪ 持有/观望", hold_count)
col4.metric("📅 数据日期", last_update)

st.divider()


# -----------------------------------------------------------------------------
# 按市场分类的 Tab 视图
# -----------------------------------------------------------------------------
st.subheader("📋 信号详情")

# 分类数据
us_results = [r for r in results if "美股" in r["市场"]]
cn_results = [r for r in results if "A股" in r["市场"]]

tabs = st.tabs(["📋 全部", "🇺🇸 美股", "🇨🇳 A股"])

with tabs[0]:
    display_signal_table(results)

with tabs[1]:
    if us_results:
        display_signal_table(us_results)
    else:
        st.info("暂无美股数据")

with tabs[2]:
    if cn_results:
        display_signal_table(cn_results)
    else:
        st.info("暂无 A 股数据")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.caption("💡 提示: 点击侧边栏「刷新数据」按钮可更新信号计算")
