# app/streamlit_app.py
# pip install streamlit pandas numpy requests
import streamlit as st, pandas as pd, numpy as np, requests, os

st.set_page_config(page_title="Finance AI Dashboard", layout="wide")
st.title("📈 Finance AI — Signals & Portfolio")

api_url = st.text_input("API base URL", value="http://localhost:8000")

tab1, tab2, tab3 = st.tabs(["Signals", "Portfolio", "Metrics"])

with tab1:
    st.subheader("Latest Signal (from API)")
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Ping /health"):
            try:
                r = requests.get(f"{api_url}/health", timeout=5)
                st.json(r.json())
            except Exception as e:
                st.warning(f"API not reachable: {e}")
    with colB:
        if st.button("Fetch /signal"):
            try:
                r = requests.get(f"{api_url}/signal", timeout=5)
                st.json(r.json())
            except Exception as e:
                st.warning(f"API not reachable: {e}")

    st.subheader("Backtest Curves")
    try:
        bt_trf = pd.read_csv("backtest_transformer.csv", index_col=0, parse_dates=True)
        df_lines = pd.DataFrame({"Market": bt_trf["Return"].cumsum(),
                                 "Transformer": bt_trf["Strat_TRF"].cumsum()})
        # เพิ่ม LSTM ถ้ามี
        if os.path.exists("backtest_lstm.csv"):
            bt_lstm = pd.read_csv("backtest_lstm.csv", index_col=0, parse_dates=True)
            df_lines["LSTM"] = bt_lstm["Strat_LSTM"].cumsum()
        st.line_chart(df_lines)
    except Exception as e:
        st.info("Put backtest_transformer.csv (and backtest_lstm.csv) in working dir.")

with tab2:
    st.subheader("Rolling Rebalance")
    try:
        rr = pd.read_csv("rolling_rebalance_returns.csv", index_col=0, parse_dates=True).iloc[:,0]
        st.line_chart(rr.cumsum())
        if os.path.exists("rolling_weights_max_sharpe.csv"):
            w = pd.read_csv("rolling_weights_max_sharpe.csv", index_col=0, parse_dates=True)
            st.caption("Latest weights (12 rebalance points)")
            st.dataframe(w.tail(12))
    except Exception as e:
        st.info("Place rolling_rebalance_returns.csv (and rolling_weights_max_sharpe.csv) in working dir.")

with tab3:
    st.subheader("Performance Metrics")
    try:
        mets = pd.read_csv("strategy_performance_metrics.csv")
        st.dataframe(mets)
    except Exception as e:
        st.info("Place strategy_performance_metrics.csv in working dir.")
