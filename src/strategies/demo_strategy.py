import polars as pl


def simple_ma_strategy(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """
    简单单均线趋势策略 (用于展示)。
    逻辑:
    - 如果 收盘价 > MA20 -> 信号: 买入
    - 如果 收盘价 < MA20 -> 信号: 卖出
    - 否则 -> 持币

    Args:
        df: 包含 close 的 Polars DataFrame
        window: 均线窗口

    Returns:
        pl.DataFrame: 包含 'signal_str' 列
    """
    # 确保按日期排序
    df = df.sort("date")

    return df.with_columns(
        [pl.col("close").rolling_mean(window).alias("ma")]
    ).with_columns(
        [
            pl.when(pl.col("close") > pl.col("ma"))
            .then(pl.lit(1))
            .when(pl.col("close") < pl.col("ma"))
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias("signal"),
            pl.when(pl.col("close") > pl.col("ma"))
            .then(pl.lit("买入"))
            .when(pl.col("close") < pl.col("ma"))
            .then(pl.lit("卖出"))
            .otherwise(pl.lit("持币"))
            .alias("signal_str"),
        ]
    )
