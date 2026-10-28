# 📉 Churn Prediction & Explainability Dashboard

End-to-end machine-learning project: data processing → feature engineering → **XGBoost** model →
**SHAP** explainability → real-time **FastAPI + Streamlit** dashboard, fully **Dockerized**.

## Highlights
- **Data & feature pipeline** – cleaning, engineered features (`avg_monthly_spend`, `charge_to_tenure_ratio`, `services_count`), scaling + one-hot encoding via a scikit-learn `ColumnTransformer`.
- **Model** – XGBoost with class-imbalance weighting and early stopping; evaluated with ROC-AUC, PR-AUC, F1.
- **Explainability** – global SHAP summary/importance plots and *per-customer* SHAP contributions returned by the API.
- **Serving** – FastAPI (`/predict`, `/predict/batch`, `/health`, `/metrics`) with Pydantic validation.
- **Dashboard** – Streamlit app with a live risk gauge, SHAP factor chart, and model insights tab.
- **Reproducible delivery** – Docker images (model trained at build time), `docker compose up`, GitHub Actions CI.

## Architecture
```
data/churn.csv ─► features.py ─► train.py ─► artifacts/{model,preprocessor}.joblib, SHAP plots, metrics.json
                                                   │
                       Streamlit (8501) ──HTTP──► FastAPI (8000) ─► ChurnPredictor (XGBoost + SHAP)
```

## Quick start

### Docker (recommended)
```bash
docker compose up --build
```
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

### Local
```bash
python -m venv .venv && source .venv/bin/activate
make install
make train        # builds data (if missing), trains model, writes artifacts/
make test
make api          # terminal 1
make dashboard    # terminal 2
```

## Example request
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "tenure": 5, "contract": "Month-to-month", "internet_service": "Fiber optic",
  "payment_method": "Electronic check", "monthly_charges": 89.5,
  "total_charges": 447.5, "num_support_calls": 3 }'
```
Response includes `churn_probability`, `risk_level`, and `top_factors` (SHAP value per feature).

## Using a real dataset
Place a CSV at `data/churn.csv` with the columns in `src/churn/config.py` (e.g. the Kaggle
*Telco Customer Churn* dataset after renaming columns) and a binary `churn` target. By default the
project generates a Telco-style synthetic dataset so it runs out of the box.

## Project layout
```
src/churn/   config, data, features, train, predictor
api/         FastAPI service
dashboard/   Streamlit app
tests/       pipeline + API tests
```

## Results
Run `make train` — metrics are written to `artifacts/metrics.json` and SHAP plots to `artifacts/`.
Add a screenshot of the dashboard here after your first run.
