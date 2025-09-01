# app/streamlit_app.py
# pip install streamlit pandas numpy requests
import os
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- Config ----------
OUT = Path("outputs")
API_DEFAULT = "http://localhost:8000"

st.set_page_config(page_title="Finance AI Dashboard", layout="wide")

# ---------- Header ----------
col_logo, col_title = st.columns([1,6])
with col_logo:
    # ใส่โลโก้ของคุณ (ถ้ามี) เช่น outputs/logo.png
    if (OUT/"logo.png").exists():
        st.image(str(OUT/"logo.png"), width=64)
with col_title:
    st.title("📈 Finance AI — Signals & Portfolio")

st.caption("LSTM/Transformer • Rolling Max-Sharpe • SHAP • Attention • FastAPI/Streamlit")

# ---------- Sidebar (Controls) ----------
st.sidebar.header("⚙️ Settings")
api_url = st.sidebar.text_input("API base URL", value=API_DEFAULT, help="เช่น http://localhost:8000")
auto_refresh = st.sidebar.checkbox("Auto-refresh (15s)", value=False)
date_from = st.sidebar.date_input("From", value=None)
date_to   = st.sidebar.date_input("To", value=None)

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Files status")
need_files = [
    "backtest_transformer.csv",
    "rolling_rebalance_returns.csv",
    "rolling_weights_max_sharpe.csv",
    "strategy_performance_metrics.csv",
]
for f in need_files:
    ok = (OUT/f).exists()
    st.sidebar.write(("✅ " if ok else "❌ ") + f)

# ---------- Helpers ----------
def load_csv(name, **kwargs):
    p = OUT / name
    if not p.exists():
        return None
    try:
        if kwargs.get("parse_dates", False):
            return pd.read_csv(p, index_col=0, parse_dates=True)
        return pd.read_csv(p)
    except Exception as e:
        st.warning(f"อ่านไฟล์ {name} ไม่ได้: {e}")
        return None

def apply_date_range(df):
    if df is None:
        return None
    if isinstance(df.index, pd.DatetimeIndex):
        if date_from:
            df = df[df.index.date >= date_from]
        if date_to:
            df = df[df.index.date <= date_to]
    return df

def kpi_card(label, value, fmt="{:.2%}"):
    if value is None or pd.isna(value):
        st.metric(label, "-")
    else:
        st.metric(label, fmt.format(value))

def try_api(path="/health"):
    try:
        r = requests.get(api_url.rstrip("/") + path, timeout=5)
        return r.status_code, r.json()
    except Exception as e:
        return None, {"error": str(e)}

# ---------- Tabs ----------
tab_signals, tab_port, tab_metrics, tab_explain = st.tabs(
    ["🔔 Signals", "💼 Portfolio", "📊 Metrics", "🧠 Explain"]
)

# =======================
# TAB 1 — Signals
# =======================
with tab_signals:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("API Status & Latest Signal")
        colA, colB, colC = st.columns(3)
        with colA:
            if st.button("Ping /health"):
                code, js = try_api("/health")
                st.code(json.dumps(js, ensure_ascii=False, indent=2))
        with colB:
            if st.button("Fetch /signal"):
                code, js = try_api("/signal")
                st.code(json.dumps(js, ensure_ascii=False, indent=2))
        with colC:
            st.write("")

    with c2:
        st.subheader("Quick Predict")
        sample = {
            "Lag1": 0.001, "Lag2": -0.0005, "Lag3": 0.0007,
            "Vol_5": 0.012, "EMA_gap": 0.002,
            "RSI14": 55, "MACD": 0.1, "MACD_sig": 0.05, "MACD_hist": 0.05,
            "Sentiment_Daily": 0.2
        }
        if st.button("POST /predict (sample)"):
            try:
                r = requests.post(api_url.rstrip("/") + "/predict",
                                  json={"data": sample}, timeout=5)
                st.code(json.dumps(r.json(), ensure_ascii=False, indent=2))
            except Exception as e:
                st.warning(f"API error: {e}")

    st.markdown("---")
    st.subheader("Backtest Curves")

    bt_trf = load_csv("backtest_transformer.csv", parse_dates=True)
    bt_lstm = load_csv("backtest_lstm.csv", parse_dates=True)
    lines = pd.DataFrame()
    if bt_trf is not None:
        bt_trf = apply_date_range(bt_trf)
        if "Return" in bt_trf.columns:
            lines["Market"] = bt_trf["Return"].cumsum()
        if "Strat_TRF" in bt_trf.columns:
            lines["Transformer"] = bt_trf["Strat_TRF"].cumsum()
    if bt_lstm is not None:
        bt_lstm = apply_date_range(bt_lstm)
        if "Strat_LSTM" in bt_lstm.columns:
            lines["LSTM"] = bt_lstm["Strat_LSTM"].cumsum()

    if not lines.empty:
        st.line_chart(lines)
        st.download_button("⬇️ Download backtest_transformer.csv",
                           data=(OUT/"backtest_transformer.csv").read_bytes(),
                           file_name="backtest_transformer.csv")
    else:
        st.info("ใส่ outputs/backtest_transformer.csv (และ backtest_lstm.csv) เพื่อแสดงกราฟ")

# =======================
# TAB 2 — Portfolio
# =======================
with tab_port:
    st.subheader("Rolling Rebalance vs EW")
    rr = load_csv("rolling_rebalance_returns.csv", parse_dates=True)
    if rr is not None:
        # รองรับทั้งไฟล์ที่มีคอลัมน์เดียวหรือหลายคอลัมน์
        if rr.shape[1] == 1:
            ser = rr.iloc[:, 0]
        else:
            # ถ้าหลายคอลัมน์ ให้เดาว่าคอลัมน์ชื่อ 'ret' หรือคอลัมน์แรก
            ser = rr["ret"] if "ret" in rr.columns else rr.iloc[:, 0]
        ser = apply_date_range(ser.to_frame()).iloc[:,0]
        st.line_chart(ser.cumsum())
        st.download_button("⬇️ Download rolling_rebalance_returns.csv",
                           data=(OUT/"rolling_rebalance_returns.csv").read_bytes(),
                           file_name="rolling_rebalance_returns.csv")
    else:
        st.info("ใส่ outputs/rolling_rebalance_returns.csv เพื่อแสดงกราฟ")

    st.markdown("### Latest Weights")
    w = load_csv("rolling_weights_max_sharpe.csv", parse_dates=True)
    if w is not None and not w.empty:
        st.dataframe(w.tail(12))
        st.download_button("⬇️ Download rolling_weights_max_sharpe.csv",
                           data=(OUT/"rolling_weights_max_sharpe.csv").read_bytes(),
                           file_name="rolling_weights_max_sharpe.csv")
    else:
        st.info("ใส่ outputs/rolling_weights_max_sharpe.csv เพื่อดูน้ำหนักล่าสุด")

# =======================
# TAB 3 — Metrics
# =======================
with tab_metrics:
    st.subheader("Performance Metrics")
    mets = load_csv("strategy_performance_metrics.csv")
    if mets is not None and not mets.empty:
        # แสดง KPI การ์ดจากแถวแรก (กลยุทธ์หลัก)
        main_row = mets.iloc[0].to_dict()
        c1, c2, c3, c4 = st.columns(4)
        kpi_card("CAGR",  main_row.get("CAGR"))
        kpi_card("Sharpe", main_row.get("Sharpe"), fmt="{:.2f}")
        kpi_card("MaxDD",  main_row.get("MaxDD"))
        kpi_card("Vol",    main_row.get("Vol"))

        st.dataframe(mets)
        st.download_button("⬇️ Download strategy_performance_metrics.csv",
                           data=(OUT/"strategy_performance_metrics.csv").read_bytes(),
                           file_name="strategy_performance_metrics.csv")
    else:
        st.info("ใส่ outputs/strategy_performance_metrics.csv เพื่อแสดงผล")

    # Optional: Rolling Sharpe ถ้ามีไฟล์ rolling_sharpe.png
    if (OUT/"rolling_sharpe.png").exists():
        st.image(str(OUT/"rolling_sharpe.png"), caption="Rolling Sharpe")

# =======================
# TAB 4 — Explain
# =======================
with tab_explain:
    st.subheader("Explainability Artifacts")
    cols = st.columns(2)
    with cols[0]:
        if (OUT/"shap_summary_xgb.png").exists():
            st.image(str(OUT/"shap_summary_xgb.png"), caption="SHAP Summary (XGB)")
        else:
            st.info("ไม่มี shap_summary_xgb.png")
    with cols[1]:
        if (OUT/"attention_heatmap.png").exists():
            st.image(str(OUT/"attention_heatmap.png"), caption="Attention Heatmap (Transformer)")
        else:
            st.info("ไม่มี attention_heatmap.png")

    # Download section
    dl_cols = st.columns(2)
    with dl_cols[0]:
        if (OUT/"shap_summary_xgb.png").exists():
            st.download_button("⬇️ SHAP PNG", data=(OUT/"shap_summary_xgb.png").read_bytes(),
                               file_name="shap_summary_xgb.png")
    with dl_cols[1]:
        if (OUT/"attention_heatmap.png").exists():
            st.download_button("⬇️ Attention PNG", data=(OUT/"attention_heatmap.png").read_bytes(),
                               file_name="attention_heatmap.png")

# ---------- Auto refresh ----------
if auto_refresh:
    st.experimental_rerun()
