# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np

# # -------------------------------
# # LOAD MODEL + SCHEMA
# # -------------------------------
# dml = joblib.load("price_churn_causal_dml.pkl")
# X_columns = joblib.load("x_columns.pkl")

# st.title("Causal Pricing Optimizer")

# st.caption(
#     "Optimal price recommendation using causal churn response "
#     "estimated via Double Machine Learning."
# )

# # -------------------------------
# # PRECOMPUTED ELASTICITIES
# # -------------------------------
# PLAN_ELASTICITY_10PCT = {
#     "Basic": 0.014283,
#     "Pro": 0.016282,
#     "Enterprise": -0.002043
# }

# INDUSTRY_ELASTICITY_10PCT = {
#     "Cybersecurity": 0.003123,
#     "DevTools": -0.008246,
#     "EdTech": 0.020692,
#     "FinTech": 0.028101,
#     "HealthTech": 0.007810
# }

# # -------------------------------
# # LOAD DATA
# # -------------------------------
# data = pd.read_csv("training_price_shock_dataset.csv")

# # -------------------------------
# # ANALYSIS LEVEL
# # -------------------------------
# analysis_level = st.selectbox(
#     "Select Analysis Level",
#     ["Plan", "Industry"]
# )

# # -------------------------------
# # SELECTION
# # -------------------------------
# if analysis_level == "Plan":
#     selection = st.selectbox(
#         "Select Plan",
#         list(PLAN_ELASTICITY_10PCT.keys())
#     )
#     elasticity_10pct = PLAN_ELASTICITY_10PCT[selection]
#     subset = data[data["plan_tier"] == selection]

# else:
#     selection = st.selectbox(
#         "Select Industry",
#         list(INDUSTRY_ELASTICITY_10PCT.keys())
#     )
#     elasticity_10pct = INDUSTRY_ELASTICITY_10PCT[selection]
#     subset = data[data["industry"] == selection]

# # -------------------------------
# # BASE METRICS
# # -------------------------------
# if subset.empty:
#     st.warning("No data available for this selection.")
#     st.stop()

# base_price = subset["mrr_amount"].mean()
# base_churn = subset["churn_flag"].mean()

# # -------------------------------
# # OPTIMIZATION
# # -------------------------------
# price_grid = np.linspace(-0.10, 0.10, 21)
# revenues = []

# for p in price_grid:
#     delta_churn = elasticity_10pct * (p / 0.10)
#     new_churn = np.clip(base_churn + delta_churn, 0, 1)
#     revenue = base_price * (1 + p) * (1 - new_churn)
#     revenues.append(revenue)

# optimal_idx = np.argmax(revenues)
# optimal_price = base_price * (1 + price_grid[optimal_idx])
# expected_revenue = revenues[optimal_idx]

# # -------------------------------
# # OUTPUT (CLEAN)
# # -------------------------------
# st.subheader("Pricing Recommendation")

# col1, col2 = st.columns(2)

# col1.metric("Base Price", f"{base_price:.2f}")
# col2.metric("Optimized Price", f"{optimal_price:.2f}")

# st.metric("Expected Revenue", f"{expected_revenue:.2f}")

# st.caption(
#     "Recommendation is derived from causal elasticity estimates "
#     "and revenue optimization under a ±10% price constraint."
# )

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pandas as pd
# import numpy as np

# # --------------------------------
# # APP SETUP
# # --------------------------------
# app = FastAPI(title="Causal Pricing API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --------------------------------
# # LOAD DATA
# # --------------------------------
# DATA = pd.read_csv("training_price_shock_dataset.csv")

# # --------------------------------
# # ELASTICITIES (+10% price change)
# # --------------------------------
# PLAN_ELASTICITY_10PCT = {
#     "Basic": 0.014283,
#     "Pro": 0.016282,
#     "Enterprise": -0.002043,
# }

# INDUSTRY_ELASTICITY_10PCT = {
#     "Cybersecurity": 0.003123,
#     "DevTools": -0.008246,
#     "EdTech": 0.020692,
#     "FinTech": 0.028101,
#     "HealthTech": 0.007810,
# }

# PRICE_GRID = np.linspace(-0.10, 0.10, 21)

# # --------------------------------
# # REQUEST / RESPONSE MODELS
# # --------------------------------
# class PricingRequest(BaseModel):
#     analysisLevel: str   # "plan" | "industry"
#     selection: str       # lowercase from UI

# class PricingResponse(BaseModel):
#     basePrice: float
#     optimizedPrice: float
#     expectedRevenue: float

# # --------------------------------
# # NORMALIZATION HELPERS
# # --------------------------------
# def normalize_level(value: str) -> str:
#     return value.strip().lower()

# def normalize_selection(value: str):
#     mapping = {
#         # Plans
#         "basic": "Basic",
#         "pro": "Pro",
#         "enterprise": "Enterprise",

#         # Industries
#         "cybersecurity": "Cybersecurity",
#         "devtools": "DevTools",
#         "edtech": "EdTech",
#         "fintech": "FinTech",
#         "healthtech": "HealthTech",
#     }
#     return mapping.get(value.strip().lower())

# # --------------------------------
# # CORE PRICING LOGIC
# # --------------------------------
# def optimize_price(analysis_level: str, selection: str):
#     level = normalize_level(analysis_level)
#     sel = normalize_selection(selection)

#     if sel is None:
#         return None

#     if level == "plan":
#         subset = DATA[DATA["plan_tier"] == sel]
#         elasticity = PLAN_ELASTICITY_10PCT.get(sel)

#     elif level == "industry":
#         subset = DATA[DATA["industry"] == sel]
#         elasticity = INDUSTRY_ELASTICITY_10PCT.get(sel)

#     else:
#         return None

#     if subset.empty or elasticity is None:
#         return None

#     base_price = float(subset["mrr_amount"].mean())
#     base_churn = float(subset["churn_flag"].mean())

#     best_price = base_price
#     best_revenue = -1.0

#     for pct in PRICE_GRID:
#         delta_churn = elasticity * (pct / 0.10)
#         new_churn = np.clip(base_churn + delta_churn, 0, 1)
#         revenue = base_price * (1 + pct) * (1 - new_churn)

#         if revenue > best_revenue:
#             best_revenue = revenue
#             best_price = base_price * (1 + pct)

#     return {
#         "basePrice": round(base_price, 2),
#         "optimizedPrice": round(best_price, 2),
#         "expectedRevenue": round(best_revenue, 2),
#     }

# # --------------------------------
# # API ENDPOINT
# # --------------------------------
# @app.post("/optimize-price", response_model=PricingResponse)
# def get_pricing(req: PricingRequest):
#     result = optimize_price(req.analysisLevel, req.selection)

#     if result is None:
#         raise HTTPException(
#             status_code=404,
#             detail="No data available for this selection"
#         )

#     return result

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pandas as pd
# import numpy as np

# # --------------------------------
# # APP SETUP
# # --------------------------------
# app = FastAPI(title="Causal Pricing API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --------------------------------
# # LOAD DATA
# # --------------------------------
# DATA = pd.read_csv("training_price_shock_dataset.csv")

# # --------------------------------
# # ELASTICITIES (+10% price change)
# # --------------------------------
# PLAN_ELASTICITY_10PCT = {
#     "Basic": 0.014283,
#     "Pro": 0.016282,
#     "Enterprise": -0.002043,
# }

# INDUSTRY_ELASTICITY_10PCT = {
#     "Cybersecurity": 0.003123,
#     "DevTools": -0.008246,
#     "EdTech": 0.020692,
#     "FinTech": 0.028101,
#     "HealthTech": 0.007810,
# }

# PRICE_GRID = np.linspace(-0.10, 0.10, 21)

# # --------------------------------
# # REQUEST / RESPONSE MODELS
# # --------------------------------
# class PricingRequest(BaseModel):
#     analysisLevel: str   # "plan" | "industry"
#     selection: str

# class PricingResponse(BaseModel):
#     basePrice: float
#     optimizedPrice: float
#     expectedRevenue: float

# class PriceChangeRequest(BaseModel):
#     analysisLevel: str   # "plan" | "industry"
#     selection: str
#     priceChangePct: float   # e.g. 5 or -3

# # --------------------------------
# # NORMALIZATION HELPERS
# # --------------------------------
# def normalize_level(value: str) -> str:
#     return value.strip().lower()

# def normalize_selection(value: str):
#     mapping = {
#         # Plans
#         "basic": "Basic",
#         "pro": "Pro",
#         "enterprise": "Enterprise",

#         # Industries
#         "cybersecurity": "Cybersecurity",
#         "devtools": "DevTools",
#         "edtech": "EdTech",
#         "fintech": "FinTech",
#         "healthtech": "HealthTech",
#     }
#     return mapping.get(value.strip().lower())

# # --------------------------------
# # CORE PRICING LOGIC (OPTIMIZATION)
# # --------------------------------
# def optimize_price(analysis_level: str, selection: str):
#     level = normalize_level(analysis_level)
#     sel = normalize_selection(selection)

#     if sel is None:
#         return None

#     if level == "plan":
#         subset = DATA[DATA["plan_tier"] == sel]
#         elasticity = PLAN_ELASTICITY_10PCT.get(sel)

#     elif level == "industry":
#         subset = DATA[DATA["industry"] == sel]
#         elasticity = INDUSTRY_ELASTICITY_10PCT.get(sel)

#     else:
#         return None

#     if subset.empty or elasticity is None:
#         return None

#     base_price = float(subset["mrr_amount"].mean())
#     base_churn = float(subset["churn_flag"].mean())

#     best_price = base_price
#     best_revenue = -1.0

#     for pct in PRICE_GRID:
#         delta_churn = elasticity * (pct / 0.10)
#         new_churn = np.clip(base_churn + delta_churn, 0, 1)
#         revenue = base_price * (1 + pct) * (1 - new_churn)

#         if revenue > best_revenue:
#             best_revenue = revenue
#             best_price = base_price * (1 + pct)

#     return {
#         "basePrice": round(base_price, 2),
#         "optimizedPrice": round(best_price, 2),
#         "expectedRevenue": round(best_revenue, 2),
#     }

# # --------------------------------
# # CORE PRICING LOGIC (SIMULATION)
# # --------------------------------
# def simulate_price_change(analysis_level: str, selection: str, price_pct: float):
#     level = normalize_level(analysis_level)
#     sel = normalize_selection(selection)

#     if sel is None:
#         return None

#     price_pct = price_pct / 100.0

#     if level == "plan":
#         elasticity = PLAN_ELASTICITY_10PCT.get(sel)

#     elif level == "industry":
#         elasticity = INDUSTRY_ELASTICITY_10PCT.get(sel)

#     else:
#         return None

#     if elasticity is None:
#         return None

#     churn_delta = elasticity * (price_pct / 0.10)

#     return round(churn_delta * 100, 3)

# # --------------------------------
# # API ENDPOINTS
# # --------------------------------
# @app.post("/optimize-price", response_model=PricingResponse)
# def get_pricing(req: PricingRequest):
#     result = optimize_price(req.analysisLevel, req.selection)

#     if result is None:
#         raise HTTPException(404, "No data available for this selection")

#     return result

# @app.post("/simulate-price-change")
# def simulate_price(req: PriceChangeRequest):
#     churn_change = simulate_price_change(
#         req.analysisLevel,
#         req.selection,
#         req.priceChangePct
#     )

#     if churn_change is None:
#         raise HTTPException(404, "Invalid selection or analysis level")

#     return {
#         "analysisLevel": req.analysisLevel,
#         "selection": req.selection,
#         "priceChangePct": req.priceChangePct,
#         "expectedChurnChangePct": churn_change
#     }

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib

# --------------------------------
# APP SETUP
# --------------------------------
app = FastAPI(title="Causal Pricing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------
# LOAD DATA + MODEL
# --------------------------------
DATA = pd.read_csv("training_price_shock_dataset.csv")

dml = joblib.load("price_churn_causal_dml.pkl")
X_columns = joblib.load("x_columns.pkl")

# --------------------------------
# ELASTICITIES (+10% price change)
# --------------------------------
PLAN_ELASTICITY_10PCT = {
    "Basic": 0.014283,
    "Pro": 0.016282,
    "Enterprise": -0.002043,
}

INDUSTRY_ELASTICITY_10PCT = {
    "Cybersecurity": 0.003123,
    "DevTools": -0.008246,
    "EdTech": 0.020692,
    "FinTech": 0.028101,
    "HealthTech": 0.007810,
}

# --------------------------------
# REQUEST MODEL
# --------------------------------
class PriceChangeRequest(BaseModel):
    analysisLevel: str               # subscription | plan | industry
    selection: Optional[str] = None  # plan / industry
    subscriptionId: Optional[str] = None
    priceChangePct: float            # e.g. 5 or -3

# --------------------------------
# HELPERS
# --------------------------------
def normalize_selection(value: str | None):
    if value is None:
        return None

    mapping = {
        "basic": "Basic",
        "pro": "Pro",
        "enterprise": "Enterprise",
        "cybersecurity": "Cybersecurity",
        "devtools": "DevTools",
        "edtech": "EdTech",
        "fintech": "FinTech",
        "healthtech": "HealthTech",
    }
    return mapping.get(value.strip().lower())

FEATURE_COLS = [
    "total_usage_count",
    "total_usage_duration_secs",
    "seats",
    "a_seats",
    "is_trial",
    "plan_tier",
    "industry",
    "country"
]

def build_X(df_raw):
    X = pd.get_dummies(
        df_raw[FEATURE_COLS],
        columns=["plan_tier", "industry", "country"],
        drop_first=True
    )

    for col in X_columns:
        if col not in X.columns:
            X[col] = 0

    return X[X_columns]

# --------------------------------
# OPTIMIZATION ENDPOINT SUPPORT
# --------------------------------
PRICE_GRID = np.linspace(-0.10, 0.10, 21)


class PricingRequest(BaseModel):
    analysisLevel: str
    selection: Optional[str] = None


class PricingResponse(BaseModel):
    basePrice: float
    optimizedPrice: float
    expectedRevenue: float


def optimize_price(analysis_level: str, selection: str | None):
    level = analysis_level.strip().lower()
    sel = normalize_selection(selection)

    if sel is None:
        return None

    if level == "plan":
        subset = DATA[DATA["plan_tier"] == sel]
        elasticity = PLAN_ELASTICITY_10PCT.get(sel)

    elif level == "industry":
        subset = DATA[DATA["industry"] == sel]
        elasticity = INDUSTRY_ELASTICITY_10PCT.get(sel)

    else:
        return None

    if subset.empty or elasticity is None:
        return None

    base_price = float(subset["mrr_amount"].mean())
    base_churn = float(subset["churn_flag"].mean())

    best_price = base_price
    best_revenue = -1.0

    for pct in PRICE_GRID:
        delta_churn = elasticity * (pct / 0.10)
        new_churn = np.clip(base_churn + delta_churn, 0, 1)
        revenue = base_price * (1 + pct) * (1 - new_churn)

        if revenue > best_revenue:
            best_revenue = revenue
            best_price = base_price * (1 + pct)

    return {
        "basePrice": round(base_price, 2),
        "optimizedPrice": round(best_price, 2),
        "expectedRevenue": round(best_revenue, 2),
    }


@app.post("/optimize-price", response_model=PricingResponse)
def get_pricing(req: PricingRequest):
    result = optimize_price(req.analysisLevel, req.selection)

    if result is None:
        raise HTTPException(404, "No data available for this selection")

    return result

# --------------------------------
# CORE SIMULATION LOGIC (MATCHES STREAMLIT)
# --------------------------------
def simulate_price_change(
    analysis_level: str,
    selection: str | None,
    subscription_id: str | None,
    price_pct: float
):
    level = analysis_level.strip().lower()
    price_pct = price_pct / 100.0

    # -------- SUBSCRIPTION (DML) --------
    if level == "subscription":
        if not subscription_id:
            return None

        row = DATA[DATA["subscription_id"] == subscription_id]
        if row.empty:
            return None

        X_input = build_X(row)
        elasticity_100pct = dml.effect(X_input)[0]

        churn_delta = elasticity_100pct * price_pct
        return round(churn_delta * 100, 2)

    # -------- PLAN --------
    if level == "plan":
        plan = normalize_selection(selection)
        elasticity = PLAN_ELASTICITY_10PCT.get(plan)
        if elasticity is None:
            return None

        churn_delta = elasticity * (price_pct / 0.10)
        return round(churn_delta * 100, 2)

    # -------- INDUSTRY --------
    if level == "industry":
        industry = normalize_selection(selection)
        elasticity = INDUSTRY_ELASTICITY_10PCT.get(industry)
        if elasticity is None:
            return None

        churn_delta = elasticity * (price_pct / 0.10)
        return round(churn_delta * 100, 2)

    return None

# --------------------------------
# API ENDPOINT
# --------------------------------
@app.post("/simulate-price-change")
def simulate_price(req: PriceChangeRequest):
    churn_change = simulate_price_change(
        req.analysisLevel,
        req.selection,
        req.subscriptionId,
        req.priceChangePct
    )

    if churn_change is None:
        raise HTTPException(404, "Invalid input or data not found")

    return {
        "analysisLevel": req.analysisLevel,
        "selection": req.selection,
        "subscriptionId": req.subscriptionId,
        "priceChangePct": req.priceChangePct,
        "expectedChurnChangePct": churn_change
    }


