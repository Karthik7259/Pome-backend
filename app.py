

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional
# import pandas as pd
# import numpy as np
# import joblib

# --------------------------------
# APP SETUP
# --------------------------------
# app = FastAPI(title="Causal Pricing API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# --------------------------------
# LOAD DATA + MODEL
# --------------------------------
# DATA = pd.read_csv("training_price_shock_dataset.csv")

# dml = joblib.load("price_churn_causal_dml.pkl")
# X_columns = joblib.load("x_columns.pkl")

# --------------------------------
# ELASTICITIES (+10% price change)
# --------------------------------
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

# --------------------------------
# REQUEST MODEL
# --------------------------------
# class PriceChangeRequest(BaseModel):
#     analysisLevel: str               
#     selection: Optional[str] = None  # plan / industry
#     subscriptionId: Optional[str] = None
#     priceChangePct: float            # e.g. 5 or -3

# --------------------------------
# HELPERS
# --------------------------------
# def normalize_selection(value: str | None):
#     if value is None:
#         return None

#     mapping = {
#         "basic": "Basic",
#         "pro": "Pro",
#         "enterprise": "Enterprise",
#         "cybersecurity": "Cybersecurity",
#         "devtools": "DevTools",
#         "edtech": "EdTech",
#         "fintech": "FinTech",
#         "healthtech": "HealthTech",
#     }
#     return mapping.get(value.strip().lower())

# FEATURE_COLS = [
#     "total_usage_count",
#     "total_usage_duration_secs",
#     "seats",
#     "a_seats",
#     "is_trial",
#     "plan_tier",
#     "industry",
#     "country"
# ]

# def build_X(df_raw):
#     X = pd.get_dummies(
#         df_raw[FEATURE_COLS],
#         columns=["plan_tier", "industry", "country"],
#         drop_first=True
#     )

#     for col in X_columns:
#         if col not in X.columns:
#             X[col] = 0

#     return X[X_columns]

# --------------------------------
# OPTIMIZATION ENDPOINT SUPPORT
# --------------------------------
# PRICE_GRID = np.linspace(-0.10, 0.10, 21)


# class PricingRequest(BaseModel):
#     analysisLevel: str
#     selection: Optional[str] = None


# class PricingResponse(BaseModel):
#     basePrice: float
#     optimizedPrice: float
#     expectedRevenue: float


# def optimize_price(analysis_level: str, selection: str | None):
#     level = analysis_level.strip().lower()
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


# @app.post("/optimize-price", response_model=PricingResponse)
# def get_pricing(req: PricingRequest):
#     result = optimize_price(req.analysisLevel, req.selection)

#     if result is None:
#         raise HTTPException(404, "No data available for this selection")

#     return result

# # --------------------------------
# # CORE SIMULATION LOGIC (MATCHES STREAMLIT)
# # --------------------------------
# def simulate_price_change(
#     analysis_level: str,
#     selection: str | None,
#     subscription_id: str | None,
#     price_pct: float
# ):
#     level = analysis_level.strip().lower()
#     price_pct = price_pct / 100.0

#     # -------- SUBSCRIPTION (DML) --------
#     if level == "subscription":
#         if not subscription_id:
#             return None

#         row = DATA[DATA["subscription_id"] == subscription_id]
#         if row.empty:
#             return None

#         X_input = build_X(row)
#         elasticity_100pct = dml.effect(X_input)[0]

#         churn_delta = elasticity_100pct * price_pct
#         return round(churn_delta * 100, 2)

#     # -------- PLAN --------
#     if level == "plan":
#         plan = normalize_selection(selection)
#         elasticity = PLAN_ELASTICITY_10PCT.get(plan)
#         if elasticity is None:
#             return None

#         churn_delta = elasticity * (price_pct / 0.10)
#         return round(churn_delta * 100, 2)

#     # -------- INDUSTRY --------
#     if level == "industry":
#         industry = normalize_selection(selection)
#         elasticity = INDUSTRY_ELASTICITY_10PCT.get(industry)
#         if elasticity is None:
#             return None

#         churn_delta = elasticity * (price_pct / 0.10)
#         return round(churn_delta * 100, 2)

#     return None

# # --------------------------------
# # API ENDPOINT
# # --------------------------------
# @app.post("/simulate-price-change")
# def simulate_price(req: PriceChangeRequest):
#     churn_change = simulate_price_change(
#         req.analysisLevel,
#         req.selection,
#         req.subscriptionId,
#         req.priceChangePct
#     )

#     if churn_change is None:
#         raise HTTPException(404, "Invalid input or data not found")

#     return {
#         "analysisLevel": req.analysisLevel,
#         "selection": req.selection,
#         "subscriptionId": req.subscriptionId,
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
# ELASTICITIES
# (ABSOLUTE churn delta for +10% price)
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

PRICE_GRID = np.linspace(-0.10, 0.10, 21)

# --------------------------------
# REQUEST MODELS
# --------------------------------
class PricingRequest(BaseModel):
    analysisLevel: str   # plan | industry
    selection: str

class PricingResponse(BaseModel):
    basePrice: float
    optimizedPrice: float
    baseRevenue: float
    expectedRevenue: float

class PriceChangeRequest(BaseModel):
    analysisLevel: str               # subscription | plan | industry
    selection: Optional[str] = None
    subscriptionId: Optional[str] = None
    priceChangePct: float            # slider value (e.g. +5, -3)

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
# PRICE OPTIMIZATION (BASE-REVENUE SAFE)
# --------------------------------
def optimize_price(analysis_level: str, selection: str):
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

    # Baseline (observed)
    base_price = float(subset["mrr_amount"].mean())
    base_churn = float(subset["churn_flag"].mean())
    base_revenue = base_price * (1 - base_churn)

    best_price = base_price
    best_revenue = base_revenue

    # Counterfactual simulation
    for pct in PRICE_GRID:
        delta_churn = elasticity * (pct / 0.10)
        expected_churn = np.clip(base_churn + delta_churn, 0, 1)
        revenue = base_price * (1 + pct) * (1 - expected_churn)

        if revenue > best_revenue:
            best_revenue = revenue
            best_price = base_price * (1 + pct)

    return {
        "basePrice": round(base_price, 2),
        "optimizedPrice": round(best_price, 2),
        "baseRevenue": round(base_revenue, 2),
        "expectedRevenue": round(best_revenue, 2),
    }

# --------------------------------
# PRICE–CHURN SIMULATION (EXPECTED CHURN)
# --------------------------------
def simulate_price_change(
    analysis_level: str,
    selection: Optional[str],
    subscription_id: Optional[str],
    price_change_pct: float
):
    level = analysis_level.strip().lower()
    price_pct = price_change_pct / 100.0

    # -------- SUBSCRIPTION (DML) --------
    if level == "subscription":
        if not subscription_id:
            return None

        row = DATA[DATA["subscription_id"] == subscription_id]
        if row.empty:
            return None

        base_churn = float(row["churn_flag"].mean())

        X_input = build_X(row)
        elasticity_100pct = dml.effect(X_input)[0]  # churn change for +100% price

        delta_churn = elasticity_100pct * price_pct
        expected_churn = np.clip(base_churn + delta_churn, 0, 1)

        return {
            "baseChurnPct": round(base_churn * 100, 2),
            "expectedChurnPct": round(expected_churn * 100, 2)
        }

    # -------- PLAN / INDUSTRY --------
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

    base_churn = float(subset["churn_flag"].mean())
    delta_churn = elasticity * (price_pct / 0.10)
    expected_churn = np.clip(base_churn + delta_churn, 0, 1)

    return {
        "baseChurnPct": round(base_churn * 100, 2),
        "expectedChurnPct": round(expected_churn * 100, 2)
    }

# --------------------------------
# API ENDPOINTS
# --------------------------------
@app.post("/optimize-price", response_model=PricingResponse)
def get_pricing(req: PricingRequest):
    result = optimize_price(req.analysisLevel, req.selection)
    if result is None:
        raise HTTPException(404, "No data available for this selection")
    return result

@app.post("/simulate-price-change")
def simulate_price(req: PriceChangeRequest):
    result = simulate_price_change(
        req.analysisLevel,
        req.selection,
        req.subscriptionId,
        req.priceChangePct
    )

    if result is None:
        raise HTTPException(404, "Invalid input or data not found")

    return {
        "analysisLevel": req.analysisLevel,
        "selection": req.selection,
        "subscriptionId": req.subscriptionId,
        "priceChangePct": req.priceChangePct,
        **result
    }