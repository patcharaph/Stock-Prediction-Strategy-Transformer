# Finance AI Project (SET50) — Starter

Quant Finance + AI project for Thai market (SET50).  
This repo scaffolds **W1–W4**: Data → Features/Labels → Baselines → LSTM, plus Portfolio (MPT).

## 📂 Structure
```
finance-ai-project/
├─ notebooks/                      # Colab notebooks
│  ├─ 01_data_ingestion.ipynb
│  ├─ 02_feature_label.ipynb
│  ├─ 03_baselines.ipynb
│  ├─ 04_lstm_benchmark.ipynb
│  └─ 05_transformer_upgrade.ipynb (coming later)
├─ portfolio/
│  ├─ 01_mpt_efficient_frontier.ipynb
│  └─ 02_cvar_analysis.ipynb (coming later)
├─ src/                           # Reusable code
│  ├─ data/ (loaders, validators)
│  ├─ features/ (technicals, labels)
│  ├─ models/ (baselines, lstm, transformer)
│  ├─ sentiment/ (th preprocess, sentiment)
│  ├─ evaluate/ (metrics, cv)
│  ├─ backtest/ (engine, rules)
│  ├─ serving/ (api, batch)
│  └─ viz/ (plots)
├─ data/            # gitignored
├─ results/         # gitignored
├─ app/             # Streamlit app
├─ tests/
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## 🚀 Open in Colab (add your username after push)
- 01 — Data Ingestion  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USERNAME>/finance-ai-project/blob/main/notebooks/01_data_ingestion.ipynb)

- 02 — Feature & Label  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USERNAME>/finance-ai-project/blob/main/notebooks/02_feature_label.ipynb)

- 03 — Baselines & Backtest  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USERNAME>/finance-ai-project/blob/main/notebooks/03_baselines.ipynb)

- 04 — LSTM Benchmark  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USERNAME>/finance-ai-project/blob/main/notebooks/04_lstm_benchmark.ipynb)

- Portfolio — MPT Frontier  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USERNAME>/finance-ai-project/blob/main/portfolio/01_mpt_efficient_frontier.ipynb)

## ▶️ Quickstart
```bash
# Python venv (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 🧭 Workflow (W1–W4)
1) **01_data_ingestion** → ดึง/ทำความสะอาด SET50 + หุ้นตัวอย่าง  
2) **02_feature_label** → ฟีเจอร์/เลเบล + time split แบบไม่รั่ว  
3) **03_baselines** → Linear Regression, XGBoost + เมตริก + backtest ง่าย ๆ  
4) **04_lstm_benchmark** → LSTM (windowed) + early stopping + backtest  
5) **Portfolio/01_mpt_efficient_frontier** → สร้าง efficient frontier + max-Sharpe

## 📝 Git Commands
```bash
git init
git remote add origin https://github.com/<USERNAME>/finance-ai-project.git
git add .
git commit -m "chore: scaffold finance-ai-project (W1–W4 starter)"
git branch -M main
git push -u origin main
```

## 📌 Notes
- `data/` และ `results/` ถูก gitignore ไว้แล้ว
- หลัง push เปลี่ยน `<USERNAME>` ใน README ให้เป็น GitHub ของคุณ

**Author:** Pae (Patchara)  
**License:** MIT