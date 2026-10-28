"""Inference + per-prediction SHAP explanations (shared by the API and tests)."""
import json

import joblib
import pandas as pd
import shap

from .config import ARTIFACT_DIR
from .features import clean_and_engineer


class ChurnPredictor:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        self.pre = joblib.load(artifact_dir / "preprocessor.joblib")
        self.model = joblib.load(artifact_dir / "model.joblib")
        self.explainer = shap.TreeExplainer(self.model)
        self.metrics = json.loads((artifact_dir / "metrics.json").read_text())

    def _transform(self, records: list[dict]) -> pd.DataFrame:
        return self.pre.transform(clean_and_engineer(pd.DataFrame(records)))

    def predict(self, records: list[dict], top_k: int = 8, threshold: float = 0.5) -> list[dict]:
        X = self._transform(records)
        proba = self.model.predict_proba(X)[:, 1]
        sv = self.explainer.shap_values(X)
        base = float(self.explainer.expected_value)
        results = []
        for i in range(len(X)):
            contrib = pd.Series(sv[i], index=X.columns)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top_k]
            results.append(
                {
                    "churn_probability": round(float(proba[i]), 4),
                    "prediction": "Churn" if proba[i] >= threshold else "Stay",
                    "risk_level": "High" if proba[i] >= 0.7 else "Medium" if proba[i] >= 0.4 else "Low",
                    "base_value": round(base, 4),
                    "top_factors": [
                        {
                            "feature": f,
                            "shap_value": round(float(v), 4),
                            "feature_value": round(float(X.iloc[i][f]), 4),
                            "direction": "increases churn risk" if v > 0 else "reduces churn risk",
                        }
                        for f, v in top.items()
                    ],
                }
            )
        return results
