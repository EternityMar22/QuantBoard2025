import streamlit as st
import polars as pl


# 设置页面配置
st.set_page_config(page_title="持仓再平衡", page_icon="⚖️", layout="wide")

st.title("⚖️ 持仓再平衡")
st.markdown("---")


# 1. 核心数据初始化
def get_initial_data():
    data = {
        "Asset": [
            "恒生科技",
            "中国可转债",
            "印度SENSEX30",
            "COMEX黄金",
            "现金",
            "A股红利指数",
            "中国国债指数",
            "比特币",
            "美国国债",
            "纳斯达克指数",
        ],
        "Target_Pct": [0.0, 28.82, 8.36, 8.86, 1.50, 20.90, 17.01, 4.70, 3.72, 6.20],
        "Current_Value": [
            0.0,
            0.0,
            30252.0,
            36489.0,
            175402.0,
            73304.0,
            59577.0,
            14183.0,
            46965.0,
            13395.0,
        ],
    }
    # 明确 Schema 确保类型安全
    schema = {"Asset": pl.Utf8, "Target_Pct": pl.Float64, "Current_Value": pl.Float64}
    return pl.DataFrame(data, schema=schema)


# 初始化 Session State
if "portfolio_df" not in st.session_state:
    st.session_state["portfolio_df"] = get_initial_data()

# 2. 界面与交互
# 侧边栏或顶部配置区
with st.container():
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        new_cash = st.number_input(
            "新增注资 (¥)",
            value=0.0,
            step=1000.0,
            format="%.2f",
            help="输入计划投入的新资金",
        )

    with col2:
        uploaded_file = st.file_uploader("导入 CSV 配置", type=["csv"])
        if uploaded_file is not None:
            try:
                # 读取二进制数据以处理潜在的编码问题
                file_bytes = uploaded_file.read()

                # 尝试解码: 优先 UTF-8 (兼容 BOM), 失败则尝试 GB18030 (Excel 默认中文编码)
                try:
                    csv_content = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    csv_content = file_bytes.decode("gb18030")

                # 使用 Polars 读取字符串内容
                # infer_schema_length=0 强制所有列读取为 String，避免推断错误，后续再清洗
                import io

                raw_df = pl.read_csv(io.StringIO(csv_content), infer_schema_length=0)

                # 简单验证列名
                required_cols = {"Asset", "Target_Pct", "Current_Value"}
                # 清洗列名 (去空格)
                raw_df.columns = [c.strip() for c in raw_df.columns]

                if not required_cols.issubset(set(raw_df.columns)):
                    st.error(f"CSV 必须包含列: {required_cols}")
                else:
                    # 转换与清洗 (Polars Expressions)
                    def clean_currency_pct(col_name):
                        return (
                            pl.col(col_name)
                            .cast(pl.Utf8)  # Ensure string
                            .str.replace_all("¥", "")
                            .str.replace_all("%", "")
                            .str.replace_all(",", "")  # Remove thousands separator
                            .str.strip_chars()  # 去除首尾空格
                            .cast(pl.Float64, strict=False)  # Safe cast
                        )

                    # 核心列清洗
                    cleaned_cols = [
                        pl.col("Asset").cast(pl.Utf8).str.strip_chars(),
                        clean_currency_pct("Target_Pct"),
                        clean_currency_pct("Current_Value"),
                    ]

                    clean_df = raw_df.select(cleaned_cols)

                    # 再次验证清洗后的数据是否为空 (cast 失败会变成 null)
                    if clean_df["Target_Pct"].is_null().all() and clean_df.height > 0:
                        st.warning("警告: 'Target_Pct' 列解析为空，请检查 CSV 格式。")

                    # 更新 Session State
                    st.session_state["portfolio_df"] = clean_df
                    st.success("配置已导入！")
            except Exception as e:
                st.error(f"导入失败 (Import Failed): {e}")

    with col3:
        # CSV 导出逻辑
        st.write("配置导出")
        # 获取当前表格 (from Session State or User Edits? Usually Session State before edits)
        # But we want to export what is currently valid.
        export_df = st.session_state["portfolio_df"]

        # 写入 CSV 字符串
        csv_str = export_df.write_csv()

        # BOM 头处理 (Excel 兼容)
        # 强制添加 BOM (\ufeff) 并编码为 utf-8 (相当于 utf-8-sig)
        csv_bytes = ("\ufeff" + csv_str).encode("utf-8")

        st.download_button(
            label="下载 CSV 配置",
            data=csv_bytes,
            file_name="portfolio_config.csv",
            mime="text/csv",
        )

st.divider()

# 3. 数据编辑与向量化计算
st.subheader("持仓配置")

# 使用 data_editor 允许用户修改基础数据
# 注意：Streamlit 返回 Pandas DataFrame，需立即转回 Polars
edited_data_pd = st.data_editor(
    st.session_state["portfolio_df"],  # Polars 原生支持
    num_rows="dynamic",
    key="portfolio_editor",
    width="stretch",
    column_config={
        "Target_Pct": st.column_config.NumberColumn(
            "目标占比 %", format="%.2f%%", min_value=0, max_value=100
        ),
        "Current_Value": st.column_config.NumberColumn("当前市值", format="¥%.0f"),
    },
)

# 立即获取编辑后的数据
# 注意: 当输入 Polars DataFrame 时，st.data_editor 现在直接返回 Polars DataFrame
edited_df = edited_data_pd  # 已经是 Polars DataFrame，无需转换

# 同步回 Session State (可选，保持状态一致性)
# st.session_state['portfolio_df'] = edited_df
# (Avoid infinite rerun loops by carefully managing logic, simple assignment here is risky if not conditional.
# But Streamlit retains editor state via 'key'. We treat edited_df as the truth for calculations.)

# 向量化计算 (Vectorized Calculations)
# 使用 lazy() 模式或直接 with_columns
# Total_Value = sum(Current_Value)
# New_Total = Total_Value + New_Cash
# ...

# Step 1: Aggregates
total_value = edited_df["Current_Value"].sum()
new_total = total_value + new_cash

# Step 2: Columnar Calcs
# 使用 with_columns 链式计算
# 注意: 如果 Total_Value 为 0，避免除零错误
result_df = edited_df.with_columns(
    [
        # Current_Pct
        (pl.col("Current_Value") / total_value * 100)
        .fill_nan(0.0)
        .alias("Current_Pct"),
        # Target_Val: 基于 New_Total 的目标市值
        (pl.lit(new_total) * pl.col("Target_Pct") / 100).alias("Target_Val"),
    ]
).with_columns(
    [
        # Adjustment
        (pl.col("Target_Val") - pl.col("Current_Value")).alias("Adjustment")
    ]
)

# 4. 混合偏差报警 (Hybrid Alert)
# 条件 A: 绝对偏差 > 5%
# 条件 B: 相对偏差 > 25% (Warning: Target_Pct could be 0)
# Status: ⚠️ or ✅

# Define conditions
cond_a = (pl.col("Current_Pct") - pl.col("Target_Pct")).abs() > 5.0
cond_b = (
    (pl.col("Current_Pct") - pl.col("Target_Pct")).abs() / pl.col("Target_Pct")
) > 0.25

# 处理 Target_Pct 为 0 导致的除零 (返回 null 或 inf)
# 只要 pl.col('Target_Pct') != 0 才检查 Cond B
# Polars expression handles nulls gracefully usually, but let's be safe.
# Status Logic
status_expr = (
    pl.when(
        cond_a | ((pl.col("Target_Pct") > 0.001) & cond_b)  # Avoid div by zero
    )
    .then(pl.lit("⚠️"))
    .otherwise(pl.lit("✅"))
    .alias("Status")
)

final_df = result_df.with_columns(status_expr)

# 5. 结果展示 (智能表格配置)
# "使用 st.data_editor 展示结果" (Read-only for calculated fields)

st.subheader("调仓计划")

st.data_editor(
    final_df,  # Polars 原生支持
    disabled=[
        "Asset",
        "Target_Pct",
        "Current_Value",
        "Target_Val",
        "Current_Pct",
        "Adjustment",
        "Status",
    ],  # Make read-only
    width="stretch",
    hide_index=True,
    column_config={
        "Asset": st.column_config.TextColumn("资产名称", width="medium"),
        "Target_Pct": st.column_config.NumberColumn("目标占比", format="%.2f%%"),
        "Current_Value": st.column_config.NumberColumn("当前市值", format="¥%d"),
        "Current_Pct": st.column_config.ProgressColumn(
            "当前占比 %",
            format="%.2f%%",
            min_value=0,
            max_value=100,
        ),
        "Target_Val": st.column_config.NumberColumn("目标市值", format="¥%d"),
        "Adjustment": st.column_config.NumberColumn(
            "调仓金额", format="¥%.0f", help="正数=买入，负数=卖出"
        ),
        "Status": st.column_config.TextColumn("状态", width="small"),
    },
)

# Summary Metrics
st.markdown("### 资产概览")
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("当前总资产", f"¥{total_value:,.0f}")
col_s2.metric("新增资金", f"¥{new_cash:,.0f}")
col_s3.metric("预估总资产", f"¥{new_total:,.0f}")
