import pandas as pd
import pytest

from churn.config import CATEGORICAL, NUMERIC, TARGET
from churn.data import generate_synthetic
from churn.features import build_preprocessor, clean_and_engineer


@pytest.fixture(scope="module")
def df():
    return generate_synthetic(n=500)


def test_synthetic_schema(df):
    assert TARGET in df.columns
    assert 0.05 < df[TARGET].mean() < 0.7


def test_feature_engineering(df):
    out = clean_and_engineer(df)
    assert {"avg_monthly_spend", "charge_to_tenure_ratio", "services_count"} <= set(out.columns)
    assert out[NUMERIC].isna().sum().sum() == 0


def test_preprocessor_outputs_named_dataframe(df):
    out = clean_and_engineer(df).drop(columns=[TARGET])
    X = build_preprocessor().fit_transform(out)
    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(df) and X.shape[1] >= len(NUMERIC) + len(CATEGORICAL)
    assert not X.isna().any().any()
