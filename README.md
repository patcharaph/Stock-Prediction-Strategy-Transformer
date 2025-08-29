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


🔲 Sentiment v1 (lexicon-based pipeline)

Tokenize ข่าว/โพสต์ภาษาไทยด้วย PyThaiNLP

Lexicon pos/neg wordlist → aggregate ต่อวัน

Join เข้ากับ features เป็น Sentiment_Daily

สร้างไฟล์ dataset_features_labels_with_sentiment.csv พร้อมเทรนใหม่ได้

⏳ สิ่งที่เหลือ (Next Steps)

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

แผนถัดไป
W5–6 (ทำต่อ)

Sentiment integration

เปิด 06_sentiment_pipeline.ipynb

เตรียม news_th.csv (หรือใช้ dummy จาก template)

รันจนได้ dataset_features_labels_with_sentiment.csv

กลับไป 05_transformer_upgrade.ipynb แล้วรันใหม่เพื่อฝึก LSTM/Transformer พร้อมฟีเจอร์ Sentiment_Daily

เป้าหมายไฟล์:

dataset_features_labels_with_sentiment.csv

metrics_transformer_test.csv, metrics_lstm_test.csv

backtest_transformer.csv, backtest_lstm.csv

Portfolio side (Rolling Rebalance + Metrics)

สร้าง/เปิด 02_mpt_rolling_rebalance.ipynb

ขั้นตอน:

โหลดราคาหุ้น (ชุดเดียวกับ MPT v1)

คำนวณผลตอบแทนรายวัน → rolling window 252d → สร้าง Max-Sharpe weights รายเดือน

ได้ซีรีส์ผลตอบแทนกลยุทธ์ (rolling_rebalance_returns.csv)

คำนวณ Sharpe, CAGR, MaxDD, Vol และเปรียบเทียบ EW

เป้าหมายไฟล์:

rolling_weights_max_sharpe.csv

rolling_rebalance_returns.csv

strategy_performance_metrics.csv

เคล็ดลับกัน error: รันโน้ตบุ๊กจากบนลงล่างในครั้งเดียวให้จบหนึ่ง workflow เพื่อเลี่ยง NameError (ตัวแปรหาย)

W7 Explainability

SHAP สำหรับ XGB / linear (เริ่มง่าย)

Attention heatmap สำหรับ Transformer (ดูว่าเน้น lag/feature ใด)

ออกไฟล์: shap_summary.png, attention_heatmap.png, และ “notes อธิบายสัญญาณ”

W8 Serving & Dashboard

FastAPI: endpoint /predict + /signal (โหลด weights/saved scaler)

Streamlit: dashboard แสดงราคา, สัญญาณ, เมตริก, น้ำหนักพอร์ต

โครงไฟล์:

src/serving/api.py (FastAPI)

app/streamlit_app.py (Streamlit)
