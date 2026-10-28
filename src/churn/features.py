"""Data cleaning + feature engineering + preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, NUMERIC

RAW_COLUMNS = [
    "gender", "senior_citizen", "partner", "dependents", "tenure", "contract",
    "internet_service", "payment_method", "paperless_billing", "tech_support",
    "online_security", "monthly_charges", "total_charges", "num_support_calls",
]


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, stateless transformations (safe to reuse at inference time)."""
    out = df.copy()

    # cleaning
    out["total_charges"] = pd.to_numeric(out["total_charges"], errors="coerce")
    out["total_charges"] = out["total_charges"].fillna(out["monthly_charges"] * out["tenure"])
    for c in ["senior_citizen", "partner", "dependents", "paperless_billing",
              "tech_support", "online_security"]:
        out[c] = out[c].astype(str).str.strip().str.title()

    # engineered features
    out["avg_monthly_spend"] = out["total_charges"] / out["tenure"].clip(lower=1)
    out["charge_to_tenure_ratio"] = out["monthly_charges"] / (out["tenure"] + 1)
    out["services_count"] = (
        (out["internet_service"] != "No").astype(int)
        + (out["tech_support"] == "Yes").astype(int)
        + (out["online_security"] == "Yes").astype(int)
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_preprocessor() -> ColumnTransformer:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")  # keeps readable feature names for SHAP
    return pre
