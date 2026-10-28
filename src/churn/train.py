"""Train the XGBoost churn model and write all artifacts.

    python -m churn.train
"""
import json

import joblib
import matplotlib
import numpy as np
import shap
from sklearn.metrics import (average_precision_score, classification_report,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .config import ARTIFACT_DIR, RANDOM_STATE, TARGET, XGB_PARAMS
from .data import load_data
from .features import build_preprocessor, clean_and_engineer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_and_engineer(load_data())
    X, y = df.drop(columns=[TARGET]), df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pre = build_preprocessor()
    Xtr, Xte = pre.fit_transform(X_train), pre.transform(X_test)

    # handle class imbalance
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw, early_stopping_rounds=30)
    model.fit(Xtr, y_train, eval_set=[(Xte, y_test)], verbose=False)

    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "best_iteration": int(model.best_iteration),
    }
    print(classification_report(y_test, pred))
    print(json.dumps(metrics, indent=2))

    # ---- global SHAP explainability ----
    explainer = shap.TreeExplainer(model)
    sample = Xte.sample(min(1000, len(Xte)), random_state=RANDOM_STATE)
    sv = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(sv, sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv, sample, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    importance = dict(
        sorted(zip(sample.columns, np.abs(sv).mean(axis=0).round(4).tolist()),
               key=lambda kv: kv[1], reverse=True)
    )
    metrics["global_shap_importance"] = importance

    joblib.dump(pre, ARTIFACT_DIR / "preprocessor.joblib")
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to {ARTIFACT_DIR}")
    return metrics


if __name__ == "__main__":
    main()
