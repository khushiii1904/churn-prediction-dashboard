"""Dataset loading. Generates a Telco-style synthetic dataset if no CSV is present.

To use the real IBM/Kaggle "Telco Customer Churn" data, map its columns to the
schema in `config.py` and save it as `data/churn.csv`.
"""
import numpy as np
import pandas as pd

from .config import DATA_PATH, RANDOM_STATE, TARGET


def generate_synthetic(n: int = 7000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
    tenure = np.where(
        contract == "Month-to-month",
        rng.integers(1, 40, n),
        rng.integers(6, 73, n),
    )
    base = np.select([internet == "DSL", internet == "Fiber optic"], [45, 80], default=20)
    monthly = np.round(base + rng.normal(0, 8, n), 2).clip(18, 120)
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2)

    df = pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], n),
            "senior_citizen": rng.choice(["Yes", "No"], n, p=[0.16, 0.84]),
            "partner": rng.choice(["Yes", "No"], n),
            "dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "tenure": tenure,
            "contract": contract,
            "internet_service": internet,
            "payment_method": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n,
                p=[0.34, 0.22, 0.22, 0.22],
            ),
            "paperless_billing": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
            "tech_support": rng.choice(["Yes", "No"], n, p=[0.35, 0.65]),
            "online_security": rng.choice(["Yes", "No"], n, p=[0.35, 0.65]),
            "monthly_charges": monthly,
            "total_charges": total,
            "num_support_calls": rng.poisson(1.5, n),
        }
    )

    logit = (
        -1.0
        + 1.3 * (df.contract == "Month-to-month")
        - 1.0 * (df.contract == "Two year")
        + 0.7 * (df.internet_service == "Fiber optic")
        + 0.5 * (df.payment_method == "Electronic check")
        + 0.3 * (df.paperless_billing == "Yes")
        + 0.4 * (df.senior_citizen == "Yes")
        - 0.6 * (df.tech_support == "Yes")
        - 0.5 * (df.online_security == "Yes")
        + 0.012 * (df.monthly_charges - 65)
        - 0.035 * df.tenure
        + 0.25 * df.num_support_calls
        + rng.normal(0, 0.6, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df[TARGET] = (rng.uniform(size=n) < prob).astype(int)
    return df


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic()
    df.to_csv(DATA_PATH, index=False)
    return df


if __name__ == "__main__":
    d = load_data()
    print(d.shape, f"churn rate = {d[TARGET].mean():.2%}")
