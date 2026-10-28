"""Central configuration: paths, column groups, model hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "churn.csv"
ARTIFACT_DIR = ROOT / "artifacts"

TARGET = "churn"
RANDOM_STATE = 42

NUMERIC = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    # engineered
    "avg_monthly_spend",
    "charge_to_tenure_ratio",
    "services_count",
]
CATEGORICAL = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "contract",
    "internet_service",
    "payment_method",
    "paperless_billing",
    "tech_support",
    "online_security",
]

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    eval_metric="auc",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
