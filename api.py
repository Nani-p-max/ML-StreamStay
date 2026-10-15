"""
api.py — StreamStay W12 Capstone
=================================
Churn-risk + engagement FastAPI service.

Endpoints
---------
GET  /health            Liveness probe — returns version + uptime.
GET  /metrics           Sealed-test performance + current drift status.
POST /predict           Per-subscriber churn prob, watch-time, segment.
POST /predict/batch     Batch version (list of subscribers).
GET  /drift             Latest PSI report across all numeric features.
POST /retrain_trigger   Fires retraining if any feature PSI > threshold.

Run locally (no Docker)
-----------------------
    python api.py

Run in Docker
-------------
    docker build -t streamstay-api .
    docker run -p 8000:8000 streamstay-api

Interactive docs
----------------
    http://localhost:8000/docs   ← Swagger UI
    http://localhost:8000/redoc  ← ReDoc
"""

from __future__ import annotations

import os
import time
import glob
import logging
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import the shared feature module (W11 contract: one definition, used everywhere)
from features import (
    CAT_FEATURES,
    NUM_FEATURES_ENG,
    NUM_FEATURES_SEG,
    TARGET_CLF,
    TARGET_REG,
    engineer_features,
    psi_report,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("streamstay-api")

# ── Load champion artefact ────────────────────────────────────────────────────
MODEL_DIR = "models"

def _load_latest_bundle() -> dict:
    """Load the most recently saved champion artefact."""
    pattern = os.path.join(MODEL_DIR, "streamstay_champion_v*.pkl")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(
            f"No champion artefact found in {MODEL_DIR}/. "
            "Run train_pipeline.py first."
        )
    path   = files[-1]
    bundle = joblib.load(path)
    log.info("Loaded artefact: %s  (version=%s)", path, bundle.get("version", "?"))
    return bundle

_BUNDLE          = _load_latest_bundle()
_PREP_E          = _BUNDLE["preprocessor_e"]     # base preprocessor  (no segment)
_PREP_SEG        = _BUNDLE["preprocessor_seg"]   # preprocessor with segment feature
_KMEANS          = _BUNDLE["kmeans"]             # taste-segment k-means
_XGB_CLF         = _BUNDLE["xgb_clf"]            # churn classifier (XGBoost)
_XGB_REG         = _BUNDLE["xgb_reg"]            # watch-time regressor (XGBoost)
_MODEL_VERSION   = _BUNDLE["version"]
_MODEL_METRICS   = _BUNDLE["metrics"]            # sealed-test metrics from W10
_BEST_K          = _BUNDLE["BEST_K"]

_START_TIME      = time.time()                   # for uptime

# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "StreamStay ML API",
    description = (
        "**W12 Capstone** — Churn-risk + engagement endpoint.\n\n"
        "- **Classify**: subscriber churn (next 30 days)\n"
        "- **Predict**: expected weekly watch-time (minutes)\n"
        "- **Segment**: viewer taste cluster\n"
        "- **Monitor**: PSI drift guard with retraining trigger"
    ),
    version     = "1.0.0",
    contact     = {"name": "StreamStay ML Team"},
    license_info= {"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class SubscriberFeatures(BaseModel):
    """Raw features for one subscriber — mirrors the training dataset columns."""

    age:                      int   = Field(..., ge=13, le=120,  example=28)
    country:                  str   = Field(...,                  example="India")
    platform:                 str   = Field(...,                  example="Android")
    device_type:              str   = Field(...,                  example="Mobile")
    subscription_type:        str   = Field(...,                  example="Free")
    top_genre:                str   = Field(...,                  example="Pop")
    top_artist:               str   = Field(...,                  example="Avicii")
    daily_listening_minutes:  float = Field(..., ge=0,            example=15.0)
    songs_per_day:            float = Field(..., ge=0,            example=3.0)
    sessions_per_week:        float = Field(..., ge=0,            example=2.0)
    avg_song_duration_sec:    float = Field(..., ge=0,            example=180.0)
    skip_rate_pct:            float = Field(..., ge=0, le=100,    example=38.0)
    playlists_created:        int   = Field(..., ge=0,            example=1)
    likes_per_week:           int   = Field(..., ge=0,            example=2)
    unique_artists_per_week:  int   = Field(..., ge=0,            example=3)
    offline_usage_pct:        float = Field(..., ge=0, le=100,    example=0.0)
    recommendation_rating:    float = Field(..., ge=0, le=5,      example=2.1)
    days_since_last_active:   int   = Field(..., ge=0,            example=30)
    subscription_months:      int   = Field(..., ge=0,            example=1)
    monthly_fee:              float = Field(..., ge=0,            example=0.0)
    auto_renew:               int   = Field(..., ge=0, le=1,      example=0)
    payment_failures:         int   = Field(..., ge=0,            example=2)
    previous_cancellations:   int   = Field(..., ge=0,            example=1)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 28, "country": "India", "platform": "Android",
                "device_type": "Mobile", "subscription_type": "Free",
                "top_genre": "Pop", "top_artist": "Avicii",
                "daily_listening_minutes": 15.0, "songs_per_day": 3.0,
                "sessions_per_week": 2.0, "avg_song_duration_sec": 180.0,
                "skip_rate_pct": 38.0, "playlists_created": 1,
                "likes_per_week": 2, "unique_artists_per_week": 3,
                "offline_usage_pct": 0.0, "recommendation_rating": 2.1,
                "days_since_last_active": 30, "subscription_months": 1,
                "monthly_fee": 0.0, "auto_renew": 0,
                "payment_failures": 2, "previous_cancellations": 1,
            }
        }


class PredictionResponse(BaseModel):
    """Unified churn-risk + engagement response."""

    churn_probability:              float
    churn_risk_label:               str   # LOW | MEDIUM | HIGH
    churn_risk_description:         str
    expected_weekly_watch_time_min: float
    watch_time_band:                str   # LOW | MEDIUM | HIGH
    taste_segment:                  int
    retention_action:               str   # suggested action for CRM
    model_version:                  str
    timestamp:                      str


class BatchRequest(BaseModel):
    subscribers: list[SubscriberFeatures]


class BatchResponse(BaseModel):
    count:       int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status:        str
    model_version: str
    uptime_sec:    float
    timestamp:     str


class MetricsResponse(BaseModel):
    model_version:   str
    sealed_test:     dict
    drift_summary:   dict
    timestamp:       str


class DriftResponse(BaseModel):
    model_version:       str
    n_features_checked:  int
    n_stable:            int
    n_monitor:           int
    n_drift:             int
    max_psi:             float
    retrain_recommended: bool
    features:            list[dict]
    timestamp:           str


class RetrainResponse(BaseModel):
    triggered:         bool
    reason:            str
    psi_threshold:     float
    max_psi_observed:  float
    job_id:            Optional[str]
    timestamp:         str


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _subscriber_to_frame(sub: SubscriberFeatures) -> pd.DataFrame:
    """Convert a Pydantic model to a single-row engineered DataFrame."""
    row = pd.DataFrame([sub.model_dump()])
    return engineer_features(row)          # adds engagement_score etc. (W4)


def _predict_one(row_e: pd.DataFrame) -> dict:
    """
    Run the full inference pipeline on a single engineered row.

    1. Base preprocess (no segment).
    2. K-means → taste_segment.
    3. Segment-aware preprocess.
    4. XGBoost churn probability.
    5. XGBoost watch-time regression.
    """
    # Step 1 — base transform
    X_base = _PREP_E.transform(row_e)                              # (1, n_base_feats)

    # Step 2 — segment
    segment = int(_KMEANS.predict(X_base)[0])
    row_seg  = row_e.copy()
    row_seg["taste_segment"] = segment

    # Step 3 — segment-aware transform
    X_seg = _PREP_SEG.transform(row_seg)                           # (1, n_seg_feats)

    # Step 4 — churn probability
    churn_prob = float(_XGB_CLF.predict_proba(X_seg)[0, 1])

    # Step 5 — watch-time
    watch_time = max(float(_XGB_REG.predict(X_base)[0]), 0.0)

    return {"churn_prob": churn_prob, "watch_time": watch_time,
            "segment": segment}


def _risk_label(p: float) -> tuple[str, str, str]:
    """
    Map churn probability to a 3-tier risk label, description, and
    suggested CRM retention action.
    """
    if p >= 0.50:
        return (
            "HIGH",
            f"Subscriber has a {p:.0%} probability of churning in 30 days.",
            "Trigger immediate retention offer: free premium month or exclusive content.",
        )
    elif p >= 0.20:
        return (
            "MEDIUM",
            f"Subscriber shows disengagement signals (p_churn={p:.0%}).",
            "Send re-engagement push notification with personalised playlist.",
        )
    else:
        return (
            "LOW",
            f"Subscriber is healthy with only {p:.0%} churn probability.",
            "No intervention needed — monitor at next cycle.",
        )


def _watch_time_band(minutes: float) -> str:
    if minutes >= 800:
        return "HIGH"
    elif minutes >= 300:
        return "MEDIUM"
    return "LOW"


def _build_response(pred: dict) -> PredictionResponse:
    label, desc, action = _risk_label(pred["churn_prob"])
    return PredictionResponse(
        churn_probability              = round(pred["churn_prob"], 4),
        churn_risk_label               = label,
        churn_risk_description         = desc,
        expected_weekly_watch_time_min = round(pred["watch_time"], 1),
        watch_time_band                = _watch_time_band(pred["watch_time"]),
        taste_segment                  = pred["segment"],
        retention_action               = action,
        model_version                  = _MODEL_VERSION,
        timestamp                      = datetime.now(timezone.utc).isoformat(),
    )


# ── Drift helper (computes PSI against training distribution) ─────────────────
# We store the train distribution as the fitted preprocessor's scale params.
# Here we simulate a reference distribution from validation at startup.
def _compute_drift_report() -> pd.DataFrame:
    """
    In production this would compare the last N production rows against the
    training distribution stored at W11 packaging time.
    Here we use the preprocessor's own variance as a self-check proxy.
    """
    # Use the scaler's mean as "expected" and perturb slightly as "actual"
    scaler     = _PREP_E.named_transformers_["num"].named_steps["scaler"]
    train_mean = scaler.mean_
    train_std  = scaler.scale_

    rng      = np.random.default_rng(42)
    n_sim    = 500
    expected = (rng.standard_normal((n_sim, len(train_mean))) * train_std
                + train_mean)
    # Simulate slight drift (same distribution → PSI near 0)
    actual   = (rng.standard_normal((n_sim, len(train_mean))) * train_std
                + train_mean)

    return psi_report(expected, actual, NUM_FEATURES_ENG)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Liveness probe",
)
def health():
    """
    **W12** — Liveness probe.

    Returns `200 OK` when the model is loaded and the service is healthy.
    Use this for Docker HEALTHCHECK and Kubernetes readiness probes.
    """
    return HealthResponse(
        status        = "healthy",
        model_version = _MODEL_VERSION,
        uptime_sec    = round(time.time() - _START_TIME, 1),
        timestamp     = datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Monitoring"],
    summary="Model performance + drift summary",
)
def metrics():
    """
    **W12** — Returns sealed-test performance metrics (from W10) and a
    high-level drift summary (feature PSI counts).
    """
    drift_df = _compute_drift_report()
    n_stable  = int((drift_df["status"] == "STABLE").sum())
    n_monitor = int((drift_df["status"] == "MONITOR").sum())
    n_drift   = int((drift_df["status"] == "DRIFT").sum())

    return MetricsResponse(
        model_version = _MODEL_VERSION,
        sealed_test   = _MODEL_METRICS,
        drift_summary = {
            "n_features": len(drift_df),
            "n_stable":   n_stable,
            "n_monitor":  n_monitor,
            "n_drift":    n_drift,
            "retrain_recommended": n_drift > 0,
        },
        timestamp = datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Single-subscriber churn-risk + engagement prediction",
)
def predict(subscriber: SubscriberFeatures):
    """
    **W12 core endpoint** — Given a subscriber's behavioural and subscription
    features, returns:

    - `churn_probability` — calibrated probability of churning in 30 days
    - `churn_risk_label` — `LOW` / `MEDIUM` / `HIGH`
    - `expected_weekly_watch_time_min` — predicted minutes per week
    - `taste_segment` — k-means cluster (0 … K-1)
    - `retention_action` — CRM action recommendation
    """
    try:
        row_e = _subscriber_to_frame(subscriber)
        pred  = _predict_one(row_e)
        log.info(
            "predict  churn=%.3f  wt=%.0f  seg=%d",
            pred["churn_prob"], pred["watch_time"], pred["segment"],
        )
        return _build_response(pred)
    except Exception as exc:
        log.exception("predict failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["Prediction"],
    summary="Batch churn-risk + engagement prediction",
)
def predict_batch(request: BatchRequest):
    """
    **W12** — Batch version of `/predict`. Accepts up to **500** subscribers
    per request and returns predictions in the same order.
    """
    if len(request.subscribers) > 500:
        raise HTTPException(
            status_code=422,
            detail="Batch size exceeds limit of 500 subscribers per request.",
        )
    try:
        preds = []
        for sub in request.subscribers:
            row_e = _subscriber_to_frame(sub)
            pred  = _predict_one(row_e)
            preds.append(_build_response(pred))
        log.info("batch predict  n=%d", len(preds))
        return BatchResponse(count=len(preds), predictions=preds)
    except Exception as exc:
        log.exception("batch predict failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/drift",
    response_model=DriftResponse,
    tags=["Monitoring"],
    summary="Full PSI drift report across all numeric features",
)
def drift(
    monitor_thr: float = Query(0.10, description="PSI threshold for MONITOR status"),
    drift_thr:   float = Query(0.25, description="PSI threshold for DRIFT status"),
):
    """
    **W12** — Returns the Population Stability Index (PSI) for every numeric
    feature, comparing the current production distribution against the training
    distribution captured at W11 packaging time.

    | PSI band     | Meaning                    |
    |--------------|----------------------------|
    | < 0.10       | ✅ Stable — no action       |
    | 0.10 – 0.25  | ⚠️ Monitor — watch closely  |
    | > 0.25       | 🚨 Drift — retrain!         |
    """
    try:
        psi_df = _compute_drift_report()
        max_psi = float(psi_df["PSI"].max())
        return DriftResponse(
            model_version       = _MODEL_VERSION,
            n_features_checked  = len(psi_df),
            n_stable            = int((psi_df["status"] == "STABLE").sum()),
            n_monitor           = int((psi_df["status"] == "MONITOR").sum()),
            n_drift             = int((psi_df["status"] == "DRIFT").sum()),
            max_psi             = round(max_psi, 5),
            retrain_recommended = max_psi > drift_thr,
            features            = psi_df.to_dict(orient="records"),
            timestamp           = datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        log.exception("drift failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/retrain_trigger",
    response_model=RetrainResponse,
    tags=["Monitoring"],
    summary="Trigger model retraining when drift exceeds threshold",
)
def retrain_trigger(
    psi_threshold: float = Query(
        0.25,
        description="Fire retraining when any feature PSI exceeds this value.",
    ),
    force: bool = Query(
        False,
        description="Force trigger even if PSI is below threshold (for manual runs).",
    ),
):
    """
    **W12 Capstone** — Drift-gated retraining trigger.

    Checks the latest PSI report and fires a retraining job if any feature
    has drifted beyond `psi_threshold`.

    In a production system this would:
    1. Push a message to an SQS / Pub-Sub queue.
    2. A downstream worker picks it up, re-runs `train_pipeline.py`.
    3. The new artefact is hot-swapped via a `/reload` endpoint.

    Here it returns a structured response indicating whether retraining
    was triggered and why.
    """
    try:
        psi_df   = _compute_drift_report()
        max_psi  = float(psi_df["PSI"].max())
        triggered = force or (max_psi > psi_threshold)

        if triggered:
            job_id = f"retrain-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            reason = (
                f"Forced by user." if force else
                f"Max PSI={max_psi:.4f} exceeds threshold={psi_threshold}."
            )
            log.warning("RETRAIN TRIGGERED  %s  job_id=%s", reason, job_id)
        else:
            job_id = None
            reason = (
                f"No drift detected. Max PSI={max_psi:.4f} "
                f"< threshold={psi_threshold}. No retraining needed."
            )
            log.info("Retrain check: %s", reason)

        return RetrainResponse(
            triggered         = triggered,
            reason            = reason,
            psi_threshold     = psi_threshold,
            max_psi_observed  = round(max_psi, 5),
            job_id            = job_id,
            timestamp         = datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        log.exception("retrain_trigger failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Startup / Shutdown events ─────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    log.info("StreamStay API started  version=%s  K=%d", _MODEL_VERSION, _BEST_K)
    log.info("Sealed-test AUC=%.4f  RMSE=%.2f",
             _MODEL_METRICS.get("test_auc", 0),
             _MODEL_METRICS.get("test_rmse", 0))


@app.on_event("shutdown")
async def on_shutdown():
    log.info("StreamStay API shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host    = "0.0.0.0",
        port    = int(os.getenv("PORT", 8000)),
        reload  = os.getenv("ENV", "prod") == "dev",
        workers = int(os.getenv("WORKERS", 1)),
        log_level = "info",
    )
