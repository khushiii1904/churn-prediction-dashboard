"""FastAPI service: real-time churn prediction with SHAP explanations."""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn import __version__
from churn.predictor import ChurnPredictor

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = ChurnPredictor()
    yield
    state.clear()


app = FastAPI(title="Churn Prediction API", version=__version__, lifespan=lifespan)

YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"] = "Male"
    senior_citizen: YesNo = "No"
    partner: YesNo = "No"
    dependents: YesNo = "No"
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    paperless_billing: YesNo = "Yes"
    tech_support: YesNo = "No"
    online_security: YesNo = "No"
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)
    num_support_calls: int = Field(ge=0, default=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "senior_citizen": "No", "partner": "Yes", "dependents": "No",
                "tenure": 5, "contract": "Month-to-month", "internet_service": "Fiber optic",
                "payment_method": "Electronic check", "paperless_billing": "Yes",
                "tech_support": "No", "online_security": "No",
                "monthly_charges": 89.5, "total_charges": 447.5, "num_support_calls": 3,
            }
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in state, "version": __version__}


@app.get("/metrics")
def metrics():
    return state["predictor"].metrics


@app.post("/predict")
def predict(customer: Customer, top_k: int = 8):
    try:
        return state["predictor"].predict([customer.model_dump()], top_k=top_k)[0]
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch")
def predict_batch(customers: list[Customer], top_k: int = 5):
    if not customers or len(customers) > 1000:
        raise HTTPException(status_code=422, detail="Send between 1 and 1000 customers.")
    return state["predictor"].predict([c.model_dump() for c in customers], top_k=top_k)
