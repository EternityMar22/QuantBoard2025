from typing import Callable
import polars as pl
import duckdb
from pathlib import Path


class BacktestEngine:
    """
    基于 Polars 的轻量级向量化回测引擎。
    """

    def __init__(self, db_path: str = "quant.db"):
        self.db_path = db_path

    def _load_data(self, ticker: str) -> pl.DataFrame:
        """
        从 DuckDB 读取特定 Ticker 的市场数据。
        """
        query = f"SELECT * FROM market_data WHERE ticker = '{ticker}' ORDER BY date"
        try:
            with duckdb.connect(self.db_path) as con:
                df = con.execute(query).pl()

            if df.is_empty():
                raise ValueError(f"No data found for ticker: {ticker}")

            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load data for {ticker}: {e}")

    def run(
        self, ticker: str, strategy_func: Callable[[pl.DataFrame], pl.DataFrame]
    ) -> pl.DataFrame:
        """
        运行回测。

        Args:
            ticker: 股票代码
            strategy_func: 接收 pl.DataFrame (OHLCV)，返回包含 'signal' 列的 pl.DataFrame。
                           signal: 1 (做多), -1 (做空), 0 (空仓)

        Returns:
            pl.DataFrame: 包含日期、价格、信号、策略收益率、净值曲线。
        """
        # 1. 加载数据
        df_raw = self._load_data(ticker)

        # 2. 执行策略
        # 策略函数必须返回包含 'signal' 列的 DataFrame
        df_strategy = strategy_func(df_raw)

        if "signal" not in df_strategy.columns:
            raise ValueError(
                "Strategy function must return a DataFrame with a 'signal' column."
            )

        # 3. 向量化计算收益
        # 假设全仓买入。
        # 逻辑: holding = signal.shift(1) (使用昨天的信号作为今天的持仓)
        # strategy_return = holding * market_return

        return (
            df_strategy.with_columns(
                [
                    # 计算市场收益率 (pct_change)
                    pl.col("close").pct_change().alias("market_return").fill_null(0.0),
                    # 确定持仓: 昨天的信号决定今天的仓位
                    pl.col("signal").shift(1).fill_null(0).alias("holding"),
                ]
            )
            .with_columns(
                [
                    # 计算策略收益
                    (pl.col("holding") * pl.col("market_return")).alias(
                        "strategy_return"
                    )
                ]
            )
            .with_columns(
                [
                    # 计算净值曲线 (Equity Curve)
                    (1 + pl.col("strategy_return")).cum_prod().alias("equity_curve")
                ]
            )
        )
