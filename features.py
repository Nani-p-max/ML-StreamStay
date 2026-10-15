"""
features.py — StreamStay Shared Feature Module
===============================================
W4  : Engineered features from viewing history, sessions, catalogue, plan.
W9  : Taste-segment feature added here so both the training pipeline and
      the API endpoint consume exactly the same definitions.
W11 : This module IS the 'shared feature module' — import it everywhere.

Usage
-----
    from features import (
        engineer_features, build_preprocessor,
        NUM_FEATURES, CAT_FEATURES, NUM_FEATURES_ENG,
        TARGET_CLF, TARGET_REG
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Column definitions ────────────────────────────────────────────────────────
TARGET_CLF = "churn_next_30_days"
TARGET_REG  = "expected_weekly_listening_minutes"

NUM_FEATURES: list[str] = [
    "age",
    "daily_listening_minutes",
    "songs_per_day",
    "sessions_per_week",
    "avg_song_duration_sec",
    "skip_rate_pct",
    "playlists_created",
    "likes_per_week",
    "unique_artists_per_week",
    "offline_usage_pct",
    "recommendation_rating",
    "days_since_last_active",
    "subscription_months",
    "monthly_fee",
    "auto_renew",
    "payment_failures",
    "previous_cancellations",
]

CAT_FEATURES: list[str] = [
    "country",
    "platform",
    "device_type",
    "subscription_type",
    "top_genre",
    "top_artist",
]

# Engineered features added by engineer_features()
ENG_FEATURES: list[str] = [
    "engagement_score",
    "loyalty_index",
    "session_density",
    "catalogue_breadth",
    "is_paid",
]

# Full numeric column set after engineering (no segment yet)
NUM_FEATURES_ENG: list[str] = NUM_FEATURES + ENG_FEATURES

# Full numeric column set with the k-means taste segment appended
NUM_FEATURES_SEG: list[str] = NUM_FEATURES_ENG + ["taste_segment"]


# ── W4: Feature engineering ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add five behavioural features derived from raw viewing-history,
    session-activity, catalogue and subscription-plan columns.

    Call this on X (no targets) before preprocessing.
    """
    out = df.copy()

    # Composite engagement score: rewards listening time, active curation and
    # positive signals; penalises skipping and recent disengagement.
    out["engagement_score"] = (
        out["daily_listening_minutes"] * 0.4
        + out["likes_per_week"]        * 5.0
        + out["playlists_created"]     * 10.0
        - out["skip_rate_pct"]         * 2.0
        - out["days_since_last_active"] * 3.0
    )

    # Loyalty index: tenure × renewal commitment minus payment friction.
    out["loyalty_index"] = (
        out["subscription_months"] * out["auto_renew"]
        - out["payment_failures"]         * 10.0
        - out["previous_cancellations"]   * 15.0
    )

    # Songs consumed per session-day (catalogue consumption density).
    out["session_density"] = (
        out["songs_per_day"] / (out["sessions_per_week"] / 7.0 + 1e-3)
    )

    # How broadly a user browses the catalogue.
    out["catalogue_breadth"] = (
        out["unique_artists_per_week"] * out["playlists_created"]
    )

    # Simple binary: is the user on a paid tier?
    out["is_paid"] = (out["monthly_fee"] > 0).astype(int)

    return out


# ── W2 / W11: Leakage-safe preprocessing factory ──────────────────────────────
def build_preprocessor(with_segment: bool = False) -> ColumnTransformer:
    """
    Return an *unfitted* ColumnTransformer.

    Parameters
    ----------
    with_segment : bool
        If True, include the 'taste_segment' column (added in W9) in the
        numeric pipeline.  Must be False when training the k-means itself.

    Notes
    -----
    Always call  preprocessor.fit(X_train)  before transforming val/test.
    Never fit on val or test — that is the leakage-safe contract.
    """
    num_cols = NUM_FEATURES_SEG if with_segment else NUM_FEATURES_ENG

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, CAT_FEATURES),
        ],
        remainder="drop",
    )


# ── Helper: get feature names after OHE ───────────────────────────────────────
def get_feature_names(fitted_preprocessor: ColumnTransformer,
                      with_segment: bool = False) -> list[str]:
    """Return the full ordered list of feature names after preprocessing."""
    ohe_names = (
        fitted_preprocessor
        .named_transformers_["cat"]
        .named_steps["ohe"]
        .get_feature_names_out(CAT_FEATURES)
        .tolist()
    )
    num_cols = NUM_FEATURES_SEG if with_segment else NUM_FEATURES_ENG
    return num_cols + ohe_names


# ── W11: PSI drift guard ───────────────────────────────────────────────────────
def compute_psi(expected: np.ndarray,
                actual: np.ndarray,
                n_bins: int = 10,
                eps: float = 1e-6) -> float:
    """
    Population Stability Index.

    Rule of thumb
    -------------
    PSI < 0.10  → stable
    0.10–0.25   → slight drift – monitor
    > 0.25      → significant drift – retrain!
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts  = np.histogram(actual,   bins=breakpoints)[0]

    exp_pct = np.clip(exp_counts / (len(expected) + eps), eps, None)
    act_pct = np.clip(act_counts  / (len(actual)   + eps), eps, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_report(X_train_mat: np.ndarray,
               X_new_mat:   np.ndarray,
               num_feature_names: list[str],
               monitor_thr: float = 0.10,
               drift_thr:   float = 0.25) -> pd.DataFrame:
    """
    Compute PSI for every numeric feature and return a tidy DataFrame
    with a status label.
    """
    rows = []
    for i, feat in enumerate(num_feature_names):
        psi = compute_psi(X_train_mat[:, i], X_new_mat[:, i])
        if psi > drift_thr:
            status = "DRIFT"
        elif psi > monitor_thr:
            status = "MONITOR"
        else:
            status = "STABLE"
        rows.append({"feature": feat, "PSI": round(psi, 5), "status": status})

    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)
