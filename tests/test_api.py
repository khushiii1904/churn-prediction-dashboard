"""Requires trained artifacts: run `make train` first."""
import pytest
from fastapi.testclient import TestClient

from churn.config import ARTIFACT_DIR

pytestmark = pytest.mark.skipif(
    not (ARTIFACT_DIR / "model.joblib").exists(), reason="artifacts not trained yet"
)

SAMPLE = {
    "gender": "Female", "senior_citizen": "No", "partner": "Yes", "dependents": "No",
    "tenure": 5, "contract": "Month-to-month", "internet_service": "Fiber optic",
    "payment_method": "Electronic check", "paperless_billing": "Yes",
    "tech_support": "No", "online_security": "No",
    "monthly_charges": 89.5, "total_charges": 447.5, "num_support_calls": 3,
}


def test_health_and_predict():
    from api.main import app
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        r = client.post("/predict", json=SAMPLE)
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["churn_probability"] <= 1
        assert len(body["top_factors"]) > 0


def test_invalid_payload_rejected():
    from api.main import app
    with TestClient(app) as client:
        assert client.post("/predict", json={**SAMPLE, "contract": "Lifetime"}).status_code == 422
