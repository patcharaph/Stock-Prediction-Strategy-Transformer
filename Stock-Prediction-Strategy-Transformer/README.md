# 📈 Stock-Prediction-Strategy-Transformer

End-to-end **Finance AI Project** สำหรับการพยากรณ์ราคาหุ้นและกลยุทธ์ลงทุน  
รวม **Quant Model (LSTM/Transformer)**, **Portfolio Optimization (MPT Rolling Rebalance)**  
และ **Serving Layer (FastAPI + Streamlit Dashboard)**

---

## 🚀 Features
- **Stock Prediction Models**
  - Baseline: Linear Regression, XGBoost
  - Deep: LSTM Benchmark, Transformer Upgrade
  - Sentiment Integration (lexicon-based daily score)
- **Portfolio Optimization**
  - Modern Portfolio Theory (MPT) Efficient Frontier
  - Rolling Rebalance (max Sharpe weights, monthly rebalance)
  - Performance metrics: Sharpe, CAGR, Max Drawdown, Volatility
- **Explainability**
  - SHAP summary for XGB
  - Attention heatmap for Transformer
- **Serving & Dashboard**
  - FastAPI endpoints: `/health`, `/predict`, `/signal`
  - Streamlit dashboard: Signals, Portfolio, Metrics

---

## 📂 Project Structure
Stock-Prediction-Strategy-Transformer/
│
├── artifacts/ # Saved models + preprocessors
│ ├── model_xgb.joblib
│ ├── model_transformer.h5
│ ├── scaler.joblib
│ └── feature_names.joblib
│
├── outputs/ # Generated outputs (backtest, metrics, plots)
│ ├── backtest_transformer.csv
│ ├── rolling_rebalance_returns.csv
│ ├── strategy_performance_metrics.csv
│ ├── shap_summary_xgb.png
│ └── attention_heatmap.png
│
├── src/serving/api.py # FastAPI serving layer
├── app/streamlit_app.py # Streamlit dashboard
├── notebooks/ # All Colab-ready notebooks (01–07)
├── requirements.txt
└── README.md


---

## 🏃 Usage

### Run API (FastAPI)

uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

streamlit run app/streamlit_app.py

📊 Example Outputs
🔹 Backtest Comparison

🔹 SHAP Summary (XGB)

🔹 Attention Heatmap (Transformer)

🔹 Streamlit Dashboard (Screenshot)

📸 Roadmap

W1–2: Data ingestion, feature/label, baselines

W3–4: LSTM benchmark, Transformer upgrade

W5–6: Sentiment integration, Rolling Rebalance + Metrics

W7: Explainability (SHAP, Attention heatmap)

W8: Serving & Dashboard (FastAPI, Streamlit)

✨ Author

Patchara Phookheaw (@patcharaph)
Finance + AI Engineering Project

