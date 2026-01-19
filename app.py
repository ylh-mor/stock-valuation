import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime, date

# 页面配置
st.set_page_config(page_title="股票估值系统 Pro", layout="wide")


# --- 核心算法 ---
def get_stock_name(ticker):
    """获取股票名称"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get("longName", info.get("shortName", ticker))
    except:  # noqa: E722
        return ticker


def analyze_stock(ticker, start_date, end_date):
    # 下载数据
    df_raw = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        interval="1d",
    )
    if df_raw.empty:
        return None

    # 获取股票名称
    stock_name = get_stock_name(ticker)

    # 处理多级索引
    prices = (
        df_raw["Close"][ticker]
        if isinstance(df_raw.columns, pd.MultiIndex)
        else df_raw["Close"]
    )
    df = pd.DataFrame({"price": prices.dropna()})

    # Log2 回归计算
    df["Time_Idx"] = np.arange(len(df))
    df["log_p"] = np.log2(df["price"])
    X = df["Time_Idx"].values.reshape(-1, 1)
    y = df["log_p"].values.reshape(-1, 1)

    model = LinearRegression().fit(X, y)
    df["trend_log"] = model.predict(X).ravel()
    sigma = np.std(df["log_p"] - df["trend_log"])

    # 指标提取
    current_price = df["price"].iloc[-1]
    curr_trend_log = df["trend_log"].iloc[-1]
    z_score = (np.log2(current_price) - curr_trend_log) / sigma

    # 年化指标
    ann_growth = (pow(2, model.coef_[0][0] * 252) - 1) * 100
    ann_vol = df["price"].pct_change().std() * np.sqrt(252) * 100
    # ann_growth = (pow(2, model.coef_[0][0] * 12) - 1) * 100
    # ann_vol = df['price'].pct_change().std() * np.sqrt(12) * 100
    # max_draw = (1 - np.exp2(np.min(df['log_p'] - df['trend_log']))) * 100
    max_bias = (1 - np.min(df["price"] / np.exp2(df["trend_log"]))) * 100
    percentile = norm.cdf(z_score) * 100

    return {
        "df": df,
        "z": z_score,
        "growth": ann_growth,
        "vol": ann_vol,
        "max_bias": max_bias,
        "price": current_price,
        "sigma": sigma,
        "trend_log": curr_trend_log,
        "percentile": percentile,
        "name": stock_name,
    }


def get_signal(z):
    if z < -1.5:
        return "💎 强力买入 (Strong Buy)", "purple"
    elif -1.5 <= z < -0.5:
        return "🟢 逢低吸纳 (Accumulate)", "green"
    elif -0.5 <= z <= 0.5:
        return "⚪ 持股观望 (Hold)", "gray"
    elif 0.5 < z <= 1.5:
        return "🟠 逢高减持 (Reduce)", "orange"
    else:
        return "🚨 强力卖出 (Strong Sell)", "red"


# --- 侧边栏控制面板 ---
with st.sidebar:
    st.title("⚙️ 系统配置")
    with st.expander("数据筛选参数", expanded=True):
        tickers_input = st.text_area(
            "输入代码 (请查询 [Yahoo Finance](https://finance.yahoo.com))",
            value="^STI,^HSI,^SPX,^NDX,000001.SS",
        )

        today = datetime.now()
        min_date = date(today.year - 50, 1, 1)  # 允许选到 50 年前

        start_date = st.date_input(
            "分析起点",
            value=date(today.year - 10, 1, 1),
            min_value=min_date,
            max_value=today,
        )
        end_date = st.date_input(
            "分析终点", value=today, min_value=start_date, max_value=today
        )

        process_btn = st.button("看估值", width="stretch")

# --- 主页面布局 ---
st.title("🚀 股票估值看板")

if process_btn:
    ticker_list = [
        t.strip().upper()
        for t in tickers_input.replace("\n", ",").split(",")
        if t.strip()
    ]
    results = {}
    summary_list = []

    with st.spinner("正在调取历史数据并计算回归..."):
        for t in ticker_list:
            data = analyze_stock(t, start_date, end_date)
            if data:
                results[t] = data
                sig, _ = get_signal(data["z"])
                summary_list.append(
                    {
                        "代码": t,
                        "名称": data["name"],
                        "当前价": f"{data['price']:.2f}",
                        "Z-Score": round(data["z"], 2),
                        "历史分位": f"{data['percentile']:.1f}%",
                        "年化增长": f"{data['growth']:.1f}%",
                        "评级": sig,
                    }
                )

    # 汇总看板 (可折叠)
    with st.expander("📊 全市场价值汇总 (Summary Table)", expanded=True):
        if summary_list:
            sum_df = pd.DataFrame(summary_list).sort_values("Z-Score")
            st.dataframe(sum_df, width="stretch", hide_index=True)

    # 详细报告 (可折叠)
    st.write("### 📑 深度个股分析报告")
    for t in ticker_list:
        if t in results:
            res = results[t]
            with st.expander(f"🔍 {t} 分析报告详情", expanded=True):
                col_info, col_chart = st.columns([1, 2.3])

                with col_info:
                    st.metric("当前股价 (Price)", f"{res['price']:.2f}")
                    st.metric("年化增长率 (Trend)", f"{res['growth']:.1f}%")
                    st.metric("当前 Z-Score", f"{res['z']:.2f}")

                    sig_text, _ = get_signal(res["z"])
                    st.info(f"**综合评级**：{sig_text}")

                    st.write("**价格参考区间**")
                    levels = [
                        ("极端泡沫(+2σ)", 2),
                        ("溢价(+1σ)", 1),
                        ("中枢", 0),
                        ("低估(-1σ)", -1),
                        ("极低估(-2σ)", -2),
                    ]
                    range_list = []
                    for label, zv in levels:
                        p = 2 ** (res["trend_log"] + zv * res["sigma"])
                        range_list.append({"位置": label, "价格": f"{p:.2f}"})
                    st.table(range_list)
                    st.caption(
                        f"波动率: {res['vol']:.1f}% | 历史分位: {res['percentile']:.1f}% | 历史最大偏离 {res['max_bias']:.1f}%"
                    )

                with col_chart:
                    # 创建交互式图表
                    fig = go.Figure()
                    df = res["df"]

                    # 添加实际价格线
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["price"],
                            mode="lines",
                            name="价格",
                            line=dict(color="rgba(0,0,0,0.3)", width=1),
                            hovertemplate="%{y:.2f}<extra></extra>",
                        )
                    )

                    # 添加趋势线
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=2 ** df["trend_log"],
                            mode="lines",
                            name="趋势中枢",
                            line=dict(color="#1E90FF", width=3),
                            hovertemplate="%{y:.2f}<extra></extra>",
                        )
                    )

                    # 添加标准差线
                    sigma_lines = [
                        (2, "red", "极端泡沫 (+2σ)"),
                        (1, "orange", "溢价 (+1σ)"),
                        (-1, "green", "低估 (-1σ)"),
                        (-2, "purple", "极低估 (-2σ)"),
                    ]

                    for sigma_val, color, name in sigma_lines:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=2 ** (df["trend_log"] + sigma_val * res["sigma"]),
                                mode="lines",
                                name=name,
                                line=dict(color=color, width=1, dash="dash"),
                                opacity=0.7,
                                hovertemplate="%{y:.2f}<extra></extra>",
                            )
                        )

                    # 设置布局
                    fig.update_layout(
                        title=dict(
                            text=f"{t} - {res['name']}",
                            font=dict(size=18, family="Arial Black"),
                            x=0.5,
                            xanchor="center",
                        ),
                        # xaxis_title="日期",
                        # yaxis_title="价格",
                        yaxis_type="log",
                        height=600,
                        width=600,
                        showlegend=False,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                        ),
                        hovermode="x unified",
                        template="plotly_white",
                    )

                    # 添加网格
                    fig.update_xaxes(
                        showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)"
                    )
                    fig.update_yaxes(
                        showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)"
                    )

                    st.plotly_chart(fig, width="stretch")

    st.info("""
        ### 风险提示 (Risk Notice)
        1. *本报告由量化脚本自动生成，不构成投资建议。*
        2. **趋势反转风险**：均值回归假设了长期增长斜率不变，若公司基本面发生根本性恶化，均值线将失效。
        3. **胖尾效应**：统计学假设是正态分布，但金融市场存在“胖尾”，即股价停留在极端区间（如 < -2σ）的时间可能远超预期。
        4. **时间成本**：回归中枢可能需要数月甚至数年，不适合极短线投机。""")

else:
    st.info("👋 请在左侧边栏配置日期跨度并点击看估值。")
