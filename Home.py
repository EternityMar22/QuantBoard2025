import streamlit as st
import duckdb
import polars as pl
from src.config import TICKERS
from src.strategies.demo_strategy import simple_ma_strategy
import os

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantBoard 2025",
    page_icon="📈",
    layout="wide",
)

st.title("QuantBoard 2025 - Signal Dashboard")


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_db_connection():
    """建立 DuckDB 连接"""
    return duckdb.connect("quant.db", read_only=True)


def get_last_update_time() -> str:
    """获取数据库中最新的数据日期"""
    try:
        with get_db_connection() as con:
            # 假设 market_data 表有 date 列
            # 检查表是否存在
            tables = con.sql("SHOW TABLES").pl()
            if "market_data" not in tables["name"].to_list():
                return "无数据"

            result = con.sql("SELECT MAX(date) FROM market_data").fetchone()
            if result and result[0]:
                return str(result[0])
            return "无数据"
    except Exception as e:
        return f"Error: {e}"


def load_ticker_data(ticker: str) -> pl.DataFrame:
    """读取指定 Ticker 的最近数据 (用于计算信号)"""
    # 读取足够多的数据以计算 MA20 + 20日涨跌幅
    # 只需读取最近 60 天即可 (假设交易日)
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
        # DuckDB DESC limit 返回的是倒序的，需要反转回时间正序以进行 rolling 计算
        return df.sort("date")
    except Exception:
        return pl.DataFrame()


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("系统状态")
last_update = get_last_update_time()
st.sidebar.info(f"上次数据更新时间:\n\n**{last_update}**")

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------

# 收集所有结果
results = []

# 展平 TICKERS 字典
all_tickers = []
for market, tickers in TICKERS.items():
    all_tickers.extend(tickers)

progress_bar = st.progress(0)
status_text = st.empty()

for i, ticker in enumerate(all_tickers):
    status_text.text(f"正在分析 {ticker} ...")
    progress_bar.progress((i + 1) / len(all_tickers))

    df_data = load_ticker_data(ticker)

    if df_data.height < 20:
        # 数据不足
        results.append(
            {
                "Ticker": ticker,
                "最新价格": None,
                "今日信号": "数据不足",
                "20日涨跌幅": None,
            }
        )
        continue

    # 运行策略
    df_processed = simple_ma_strategy(df_data)

    # 获取最新一行的结果
    latest_row = df_processed.tail(1)
    current_price = latest_row["close"][0]
    signal = latest_row["signal_str"][0]

    # 计算20日涨跌幅
    # 需要往前找20个交易日 (row - 20)
    # 注意: df_data 长度可能不够长，虽然前面做了检查
    pct_chg_20d = 0.0
    if df_processed.height >= 21:
        # 索引 -1 是最新，索引 -21 是20天前
        price_20d_ago = df_processed["close"][-21]
        if price_20d_ago != 0:
            pct_chg_20d = (current_price - price_20d_ago) / price_20d_ago

    results.append(
        {
            "Ticker": ticker,
            "最新价格": current_price,
            "今日信号": signal,
            "20日涨跌幅": pct_chg_20d,
        }
    )

progress_bar.empty()
status_text.empty()

# -----------------------------------------------------------------------------
# Display Dataframe
# -----------------------------------------------------------------------------
if results:
    df_results = pl.DataFrame(results)

    # 格式化
    # Polars 到 Pandas 转换以使用 st.dataframe 的 styler (Streamlit 原生支持 Pandas Styler)
    # 或者直接使用 st.dataframe 的 column_config

    # 转换为 Pandas for display (Streamlit 兼容性更好)
    df_display = df_results.to_pandas()

    # 样式高亮函数
    def highlight_buy(row):
        return [
            "background-color: #d4edda; color: #155724"
            if row["今日信号"] == "买入"
            else ""
            for _ in row
        ]

    st.subheader("市场信号概览")

    st.dataframe(
        df_display.style.apply(highlight_buy, axis=1).format(
            {"最新价格": "{:.2f}", "20日涨跌幅": "{:.2%}"}
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("标的代码"),
            "最新价格": st.column_config.NumberColumn("最新价格", format="%.2f"),
            "今日信号": st.column_config.TextColumn("今日信号"),
            "20日涨跌幅": st.column_config.NumberColumn("20日涨跌幅", format="%.2%"),
        },
    )
else:
    st.warning("暂无数据显示。请检查数据库是否已填充。")
