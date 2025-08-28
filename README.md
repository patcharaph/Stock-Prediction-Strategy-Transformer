# Stock-Prediction-Strategy-Transformer
Stock Market Prediction &amp; Trading Strategy (Quant AI) &amp; Portfolio Optimization with AI

✅ สิ่งที่ทำเสร็จแล้ว (W1–2 → W3–4)
🔹 W1–2: Data + Baselines

✔️ Data ingestion (01_data_ingestion.ipynb)

โหลด SET50 Index + หุ้นตัวอย่าง (PTT, AOT, SCB, CPALL, ADVANC)

Clean/forward-fill missing

เซฟเป็น idx_clean.csv, prices_clean.csv

✔️ Feature & Label (02_feature_label.ipynb)

ฟีเจอร์: Returns, Lags, Volatility, EMA10/20, EMA gap

Label: Next-day return

Train/Val/Test split แบบไม่รั่วเวลา

เซฟเป็น dataset_features_labels.csv

✔️ Baselines (LR & XGB) (03_baselines.ipynb)

เทรน Linear Regression + XGBoost

Metrics: RMSE, MAE, R², Directional Accuracy

Backtest baseline signals

Portfolio prep: mean_returns, cov_matrix

🔹 W3–4: Deep Models & Sentiment

✔️ LSTM benchmark (04_lstm_benchmark.ipynb)

ใช้ sliding window (20 วัน)

Early stopping

Metrics + backtest เทียบตลาด

✔️ Transformer baseline (05_transformer_upgrade.ipynb – draft code ready)

Encoder block (MultiHeadAttention + FFN)

เทียบ metrics กับ LSTM

Backtest กราฟเทียบ LSTM vs Transformer vs Market


⏳ สิ่งที่เหลือ (Next Steps)

🔲 Sentiment v1 (lexicon-based pipeline)

Tokenize ข่าว/โพสต์ภาษาไทยด้วย PyThaiNLP

Lexicon pos/neg wordlist → aggregate ต่อวัน

Join เข้ากับ features เป็น Sentiment_Daily

สร้างไฟล์ dataset_features_labels_with_sentiment.csv พร้อมเทรนใหม่ได้

🔲 MPT Efficient Frontier v1 (01_mpt_efficient_frontier.ipynb)

ใช้ Ledoit–Wolf covariance

สร้าง efficient frontier

หา Max-Sharpe portfolio

Export mpt_weights_max_sharpe.csv

🔲 Rolling Rebalance baseline

Monthly rebalance (ใช้ lookback 252 วัน)

Generate rolling_weights_max_sharpe.csv + กราฟผลตอบแทนเทียบ Equal-Weight baseline


A) Deep Models

🔲 โน้ตบุ๊ก 05_transformer_upgrade.ipynb

รวม Transformer + Compare table (LSTM vs Transformer)

ใส่ Attention visualization (heatmap ดูว่าโมเดลโฟกัส lag ไหน)

🔲 โน้ตบุ๊ก 06_sentiment_pipeline.ipynb

จัดเต็ม Sentiment pipeline (เก็บข่าวจริง/FB → tokenize → sentiment model → join features)

ปัจจุบัน v1 ยังเป็น lexicon (เบาๆ) → เหลืออัปเกรดเป็น WangchanBERTa / finetuned model

B) Portfolio

🔲 โน้ตบุ๊ก 02_mpt_rolling_rebalance.ipynb

ทำ rolling rebalance ให้ครบ (รายเดือน/รายสัปดาห์)

คำนวณ performance metrics: CAGR, Volatility, Sharpe, Max Drawdown

🔲 โน้ตบุ๊ก 03_drl_env_training.ipynb

Deep Reinforcement Learning (Stable Baselines3)

สร้าง environment → เทรน PPO/DDPG → เปรียบเทียบกับ MPT baseline

C) Integration

🔲 Backtest engine (src/backtest/) → ทำฟังก์ชัน reusable

🔲 Explainability notebook → ใช้ SHAP, Attention, trade narrative

🔲 Serving (FastAPI) → สร้าง API เสิร์ฟสัญญาณประจำวัน

🔲 Dashboard (Streamlit) → แสดงสัญญาณ + attribution + portfolio weights

🚦 สถานะปัจจุบัน

📍 เราอยู่ ปลาย W3–4

Baselines → ✅

LSTM → ✅

Transformer → ✅ (ยังต้อง wrap เป็น notebook + add Attention)

Sentiment v1 → ✅ (ยังไม่เป็น production-ready sentiment model)

MPT Frontier → ✅

Rolling Rebalance → ✅ (baseline เวอร์ชันแรก)
