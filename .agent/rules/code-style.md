---
trigger: always_on
---

定义:Python 高频交易系统专家。
核心规则：
1.优先使用 polars 进行数据处理，严禁使用 pandas 循环。
2.数据库操作必须使用 duckdb 的上下文管理器，防止锁死。
3.代码风格要求 Python 3.12+，使用 Type Hints。
4.前端使用 streamlit，不要写任何 HTML/CSS。

目录结构说明：
QuantBoard2025/
├── .env                # 存放 Token (TG_TOKEN, TUSHARE_TOKEN)
├── daily_runner.py     # [后端] 每日任务入口 (ETL -> Strategy -> Alert)
├── Home.py             # [前端] Streamlit 首页 (信号概览)
├── quant.db            # [数据] DuckDB 单文件数据库
├── pyproject.toml      # [配置] 依赖管理
├── pages/              # [前端] Streamlit 分页面
│   └── 01_Portfolio.py # 持仓管理页
└── src/                # [逻辑核心]
    ├── config.py       # 策略注册表、常量定义
    ├── data_loader.py  # 负责下载并存入 DuckDB
    ├── engine.py       # 纯 Polars 的向量化回测引擎
    └── strategies/     # 策略仓库
        ├── ma_cross.py # 具体策略文件
        └── ...