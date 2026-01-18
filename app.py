import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, date

# 页面配置
st.set_page_config(page_title="股票估值系统 Pro", layout="wide")

# --- 核心算法 ---
def get_stock_name(ticker):
    """获取股票名称"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('longName', info.get('shortName', ticker))
    except:
        return ticker

def analyze_stock(ticker, start_date, end_date):
    # 下载数据
    df_raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, interval='1d')
    if df_raw.empty: return None

    # 获取股票名称
    stock_name = get_stock_name(ticker)

    # 处理多级索引
    prices = df_raw['Close'][ticker] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw['Close']
    df = pd.DataFrame({'price': prices.dropna()})
    
    # Log2 回归计算
    df['Time_Idx'] = np.arange(len(df))
    df['log_p'] = np.log2(df['price'])
    X = df['Time_Idx'].values.reshape(-1, 1)
    y = df['log_p'].values.reshape(-1, 1)
    
    model = LinearRegression().fit(X, y)
    df['trend_log'] = model.predict(X).ravel()
    sigma = np.std(df['log_p'] - df['trend_log'])
    
    # 指标提取
    current_price = df['price'].iloc[-1]
    curr_trend_log = df['trend_log'].iloc[-1]
    z_score = (np.log2(current_price) - curr_trend_log) / sigma
    
    # 年化指标
    ann_growth = (pow(2, model.coef_[0][0] * 252) - 1) * 100
    ann_vol = df['price'].pct_change().std() * np.sqrt(252) * 100
    # ann_growth = (pow(2, model.coef_[0][0] * 12) - 1) * 100
    # ann_vol = df['price'].pct_change().std() * np.sqrt(12) * 100
    percentile = norm.cdf(z_score) * 100

    return {
        "df": df, "z": z_score, "growth": ann_growth, "vol": ann_vol, 
        "price": current_price, "sigma": sigma, "trend_log": curr_trend_log,
        "percentile": percentile, "name": stock_name
    }

def get_signal(z):
    if z < -1.5: return "💎 强力买入", "purple"
    elif -1.5 <= z < -0.5: return "🟢 逢低吸纳", "green"
    elif -0.5 <= z <= 0.5: return "⚪ 持股观望", "gray"
    elif 0.5 < z <= 1.5: return "🟠 逢高减持", "orange"
    else: return "🚨 强力卖出", "red"

# --- 侧边栏控制面板 ---
with st.sidebar:
    st.title("⚙️ 系统配置")
    with st.expander("数据筛选参数", expanded=True):
        tickers_input = st.text_area("输入代码 (请查询 [Yahoo Finance](https://finance.yahoo.com))", value="^STI,^HSI,^SPX,^NDX,000001.SS")
        
        # 恢复使用 date_input 并支持 50 年范围
        today = datetime.now()
        min_date = date(today.year - 50, 1, 1) # 允许选到 50 年前
        
        start_date = st.date_input("分析起点", value=date(today.year - 10, 1, 1), min_value=min_date)
        end_date = st.date_input("分析终点", value=today, min_value=min_date)
        
        process_btn = st.button("开始量化巡检", use_container_width=True)

# --- 主页面布局 ---
st.title("🚀 股票估值看板")

if process_btn:
    ticker_list = [t.strip().upper() for t in tickers_input.replace('\n', ',').split(',') if t.strip()]
    results = {}
    summary_list = []

    with st.spinner('正在调取历史数据并计算回归...'):
        for t in ticker_list:
            data = analyze_stock(t, start_date, end_date)
            if data:
                results[t] = data
                sig, _ = get_signal(data['z'])
                summary_list.append({
                    "代码": t, "名称": data['name'], "当前价": f"{data['price']:.2f}", "Z-Score": round(data['z'], 2), 
                    "历史分位": f"{data['percentile']:.1f}%", "年化增长": f"{data['growth']:.1f}%", "评级": sig
                })

    # 汇总看板 (可折叠)
    with st.expander("📊 全市场价值汇总 (Summary Table)", expanded=True):
        if summary_list:
            sum_df = pd.DataFrame(summary_list).sort_values("Z-Score")
            st.dataframe(sum_df, use_container_width=True, hide_index=True)
    
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
                    
                    sig_text, _ = get_signal(res['z'])
                    st.info(f"**综合评级**：{sig_text}")
                    
                    st.write("**价格参考区间**")
                    levels = [
                        ("极端泡沫(+2σ)", 2), ("溢价(+1σ)", 1), 
                        ("中枢(0σ)", 0), ("低估(-1σ)", -1), ("极低估(-2σ)", -2)
                    ]
                    range_list = []
                    for label, zv in levels:
                        p = 2**(res['trend_log'] + zv * res['sigma'])
                        range_list.append({"位置": label, "价格": f"{p:.2f}"})
                    st.table(range_list)
                    st.caption(f"波动率: {res['vol']:.1f}% | 历史分位: {res['percentile']:.1f}%")

                with col_chart:
                    # 绘图逻辑：无 Legend，极简标题
                    fig, ax = plt.subplots(figsize=(12, 7))
                    df = res['df']
                    ax.plot(df.index, df['price'], color='black', alpha=0.15, linewidth=1)
                    ax.plot(df.index, 2**df['trend_log'], color='#1E90FF', lw=2.5)
                    
                    # 绘制 4 条标准差线
                    ax.plot(df.index, 2**(df['trend_log'] + 2*res['sigma']), color='red', ls='--', alpha=0.7)
                    ax.plot(df.index, 2**(df['trend_log'] + 1*res['sigma']), color='orange', ls='--', alpha=0.7)
                    ax.plot(df.index, 2**(df['trend_log'] - 1*res['sigma']), color='green', ls='--', alpha=0.7)
                    ax.plot(df.index, 2**(df['trend_log'] - 2*res['sigma']), color='purple', ls='--', alpha=0.7)
                    
                    ax.set_yscale('log', base=2)
                    ax.set_title(f"{t} - {res['name']}", fontsize=18, fontweight='bold')
                    ax.grid(True, which='both', alpha=0.1)
                    st.pyplot(fig)
            
else:
    st.info("👋 请在左侧侧边栏配置日期跨度并点击开始巡检。")
