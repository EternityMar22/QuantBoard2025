import os
from typing import Dict, List, Callable, Any, Final
import polars as pl

# -----------------------------------------------------------------------------
# 1. 标的配置 (TICKERS)
# -----------------------------------------------------------------------------
# 定义要监控的投资标的，按市场/类别分类
TICKERS: Final[Dict[str, List[str]]] = {
    "US": ["SPY", "GLD", "NVDA"],  # 美股/ETF
    "CN": ["600519", "510300"],  # A股 (贵州茅台, 沪深300ETF)
}

# -----------------------------------------------------------------------------
# 2. 策略注册表 (STRATEGY_REGISTRY)
# -----------------------------------------------------------------------------
# 存储所有可用策略的函数对象。
# 键为策略名称 (str)，值为策略函数 (Callable)。
# 策略函数签名预期: func(df: pl.DataFrame, **kwargs) -> pl.DataFrame
# 目前初始化为空，后续在 strategies/__init__.py 或具体策略文件中注册
STRATEGY_REGISTRY: Dict[str, Callable[..., pl.DataFrame]] = {}

# -----------------------------------------------------------------------------
# 3. 环境配置 (Environment Variables)
# -----------------------------------------------------------------------------
# 尝试读取 Telegram Bot Token
# 建议在项目根目录 .env 文件中配置: TG_TOKEN=your_token_here
TG_TOKEN: Final[str] = os.getenv("TG_TOKEN", "")

if not TG_TOKEN:
    # 仅作为警告打印，不阻断程序运行 (可能只跑回测不需要 Bot)
    print("警告：未在环境变量中找到 'TG_TOKEN'。Telegram 提醒功能将被禁用。")
