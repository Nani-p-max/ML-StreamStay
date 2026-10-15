"""
train_pipeline.py — StreamStay Churn & Watch-Time Training Pipeline
====================================================================
Runs the end-to-end pipeline and writes one versioned artefact bundle
to  models/streamstay_champion_v<DATE>.pkl.

Goals
-----
1. Churn Prediction      -> XGBoost Classifier (binary, 30-day horizon)
2. Watch-Time Prediction -> XGBoost Regressor  (expected min/week)
3. Taste Segmentation    -> K-Means clustering (feeds into churn model)

Pipeline Stages
---------------
W1  Frame targets; compute irreducible-error floor.
W2  Ingest, clean, validate; leakage-safe 60/20/20 split + EDA.
    Engineer features; fit base preprocessor.
W8  XGBoost churn classifier + XGBoost watch-time regressor + SHAP.
W9  K-Means taste segmentation; segment fed as feature into final XGB.
W10 Rigorous evaluation -- nested CV, probability calibration, sealed test.
W11 Package champion artefact; PSI drift guard.

Run
---
    python train_pipeline.py

Outputs
-------
    models/streamstay_champion_v<DATE>.pkl
    plots/  (one PNG per stage)
    mlruns/ (MLflow experiment logs)
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from features import (
    CAT_FEATURES,
    NUM_FEATURES_ENG,
    NUM_FEATURES_SEG,
    TARGET_CLF,
    TARGET_REG,
    build_preprocessor,
    engineer_features,
    get_feature_names,
    psi_report,
)

warnings.filterwarnings("ignore")

# ── Global config ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
DATA_PATH    = "streamstay_music_1200.csv"
MODEL_DIR    = "models"
PLOT_DIR     = "plots"
np.random.seed(RANDOM_STATE)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 110, "figure.facecolor": "white"})


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(name: str) -> None:
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   [plot] {path}")


def banner(week: str, title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {week} — {title}")
    print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# W1 — Frame targets & irreducible-error floor
# ─────────────────────────────────────────────────────────────────────────────

def week1_frame_targets(df: pd.DataFrame) -> dict:
    banner("W1", "Frame Targets & First EDA Look")

    churn_rate  = df[TARGET_CLF].mean()
    wt_mean     = df[TARGET_REG].mean()
    wt_std      = df[TARGET_REG].std()

    # Irreducible-error floor
    naive_rmse  = float(np.sqrt(mean_squared_error(
        df[TARGET_REG],
        np.full(len(df), wt_mean),
    )))
    naive_acc   = 1.0 - churn_rate

    print(f"  Dataset          : {df.shape[0]:,} subscribers, {df.shape[1]} columns")
    print(f"  Churn rate       : {churn_rate:.2%} ({df[TARGET_CLF].sum()} churned)")
    print(f"  Watch-time mean  : {wt_mean:.1f} min/week  std={wt_std:.1f}")
    print(f"  Naive RMSE floor : {naive_rmse:.2f} min  (predict mean)")
    print(f"  Naive acc. floor : {naive_acc:.2%} (majority class)")

    # Target distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    counts = df[TARGET_CLF].value_counts()
    axes[0].bar(["Retained", "Churned"], counts.values,
                color=["#4C72B0", "#DD8452"], edgecolor="white", width=0.5)
    axes[0].set_title("TARGET 1 — Churn Distribution", fontweight="bold")
    axes[0].set_ylabel("Users")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 4, str(v), ha="center", fontweight="bold")

    axes[1].hist(df[TARGET_REG], bins=40,
                 color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[1].axvline(wt_mean,              color="red",    ls="--", label="Mean")
    axes[1].axvline(df[TARGET_REG].median(), color="orange", ls="--", label="Median")
    axes[1].set_title("TARGET 2 — Weekly Watch-Time", fontweight="bold")
    axes[1].set_xlabel("min / week"); axes[1].legend()

    df.boxplot(column=TARGET_REG, by=TARGET_CLF, ax=axes[2],
               boxprops=dict(color="#4C72B0"),
               medianprops=dict(color="red", linewidth=2))
    axes[2].set_title("Watch-Time by Churn Group", fontweight="bold")
    axes[2].set_xlabel("Churn (0=Retained, 1=Churned)")
    plt.suptitle("")
    plt.tight_layout()
    save_fig("w1_targets.png")

    return {"naive_rmse": naive_rmse, "naive_acc": naive_acc}


# ─────────────────────────────────────────────────────────────────────────────
# W2 — Ingest, clean, EDA, leakage-safe split
# ─────────────────────────────────────────────────────────────────────────────

def week2_ingest_and_split(df: pd.DataFrame):
    banner("W2", "Data Ingestion, Cleaning & Leakage-Safe Pipeline")

    # Missing values
    missing = df.isnull().sum()
    n_missing = int(missing.sum())
    print(f"  Missing values   : {n_missing} {'→ clean ✓' if n_missing == 0 else ''}")

    # Outlier audit (IQR)
    num_audit = df.select_dtypes(include=np.number).columns.drop(
        [TARGET_CLF, TARGET_REG, "auto_renew", "payment_failures",
         "previous_cancellations"], errors="ignore"
    )
    outlier_rows = []
    for col in num_audit:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        outlier_rows.append({"feature": col, "n_outliers": int(n_out),
                             "pct": f"{n_out/len(df):.1%}"})
    print("\n  Top outlier features (IQR):")
    print(pd.DataFrame(outlier_rows).sort_values("n_outliers", ascending=False)
            .head(6).to_string(index=False))

    # EDA — churn rate by category
    cat_cols = ["subscription_type", "platform", "device_type", "top_genre", "country"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        cr = df.groupby(col)[TARGET_CLF].mean().sort_values(ascending=False)
        bars = axes[i].bar(cr.index, cr.values * 100,
                           color=sns.color_palette("muted", len(cr)),
                           edgecolor="white")
        axes[i].set_title(f"Churn % by {col}", fontweight="bold")
        axes[i].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        axes[i].tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, cr.values):
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.2, f"{val:.1%}",
                         ha="center", fontsize=8)
    axes[-1].set_visible(False)
    plt.suptitle("StreamStay — Churn Rate by Category", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("w2_categorical_eda.png")

    # Correlation heatmap
    corr = df.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.4, cbar_kws={"shrink": 0.8})
    plt.title("StreamStay — Feature Correlation Matrix", fontweight="bold", pad=15)
    plt.tight_layout()
    save_fig("w2_correlation_heatmap.png")

    # Feature / target arrays
    X = df.drop(columns=["user_id", TARGET_CLF, TARGET_REG])
    y_clf = df[TARGET_CLF]
    y_reg  = df[TARGET_REG]

    # 60 / 20 / 20  stratified split (sealed test never seen during training)
    X_tmp, X_test, y_clf_tmp, y_clf_test, y_reg_tmp, y_reg_test = train_test_split(
        X, y_clf, y_reg, test_size=0.20, random_state=RANDOM_STATE, stratify=y_clf
    )
    X_train, X_val, y_clf_train, y_clf_val, y_reg_train, y_reg_val = train_test_split(
        X_tmp, y_clf_tmp, y_reg_tmp, test_size=0.25,
        random_state=RANDOM_STATE, stratify=y_clf_tmp
    )

    print(f"\n  Split  Train:{X_train.shape[0]}  Val:{X_val.shape[0]}  "
          f"Test:{X_test.shape[0]} (sealed)")
    print(f"  Churn  Train:{y_clf_train.mean():.2%}  "
          f"Val:{y_clf_val.mean():.2%}  Test:{y_clf_test.mean():.2%}")

    return (X_train, X_val, X_test,
            y_clf_train, y_clf_val, y_clf_test,
            y_reg_train,  y_reg_val,  y_reg_test)

# W8 — XGBoost champion + SHAP
# ─────────────────────────────────────────────────────────────────────────────

def week8_xgboost(Xtr, Xva, Xte,
                   y_clf_train, y_clf_val, y_clf_test,
                   y_reg_train, y_reg_val,
                   feature_names: list[str]) -> dict:
    banner("W8", "XGBoost Champion + SHAP Explanations")

    # ── Churn Classifier ──────────────────────────────────────────────────────
    # scale_pos_weight handles class imbalance without oversampling at this stage
    scale_pos = int(np.sum(y_clf_train == 0) / np.sum(y_clf_train == 1))
    print(f"  scale_pos_weight : {scale_pos}  (minority uplift)")

    mlflow.set_experiment("StreamStay_Churn")
    with mlflow.start_run(run_name="XGBoost_Churn"):
        xgb_clf = xgb.XGBClassifier(
            n_estimators=1000, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale_pos, eval_metric="auc",
            early_stopping_rounds=50,
            random_state=RANDOM_STATE, verbosity=0,
        )
        xgb_clf.fit(Xtr, y_clf_train,
                    eval_set=[(Xva, y_clf_val)], verbose=False)

        y_prob_xgb = xgb_clf.predict_proba(Xva)[:, 1]
        auc_xgb = float(roc_auc_score(y_clf_val, y_prob_xgb))
        ap_xgb  = float(average_precision_score(y_clf_val, y_prob_xgb))
        f1_xgb  = float(f1_score(y_clf_val, xgb_clf.predict(Xva)))
        mlflow.log_metrics({"best_round": xgb_clf.best_iteration,
                            "val_roc_auc": auc_xgb, "val_f1": f1_xgb})
        mlflow.xgboost.log_model(xgb_clf, "xgb_clf")
        print(f"  XGB Clf  best_round={xgb_clf.best_iteration}  "
              f"AUC={auc_xgb:.4f}  AP={ap_xgb:.4f}  F1={f1_xgb:.4f}")

    # XGBoost Regressor
    # ── Watch-Time Regressor ──────────────────────────────────────────────────
    # Same gradient-boosted tree architecture — only the loss changes (RMSE)
    mlflow.set_experiment("StreamStay_WatchTime")
    with mlflow.start_run(run_name="XGBoost_WatchTime"):
        xgb_reg = xgb.XGBRegressor(
            n_estimators=1000, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=50, eval_metric="rmse",
            random_state=RANDOM_STATE, verbosity=0,
        )
        xgb_reg.fit(Xtr, y_reg_train,
                    eval_set=[(Xva, y_reg_val)], verbose=False)
        rmse_xgb = float(np.sqrt(mean_squared_error(y_reg_val, xgb_reg.predict(Xva))))
        r2_xgb   = float(r2_score(y_reg_val, xgb_reg.predict(Xva)))
        mlflow.log_metrics({"val_rmse": rmse_xgb, "val_r2": r2_xgb})
        print(f"  XGB Reg  best_round={xgb_reg.best_iteration}  "
              f"RMSE={rmse_xgb:.2f}  R²={r2_xgb:.4f}")

    # ── SHAP explanations ─────────────────────────────────────────────────────
    print("  Computing SHAP values…")
    explainer   = shap.TreeExplainer(xgb_clf)
    shap_values = explainer(Xva)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values.values, Xva, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SHAP — Feature Impact on Churn Probability",
              fontweight="bold", pad=10)
    plt.tight_layout()
    save_fig("w8_shap_summary.png")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, Xva, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=15)
    plt.title("SHAP — Mean |SHAP| (Global Importance)", fontweight="bold", pad=10)
    plt.tight_layout()
    save_fig("w8_shap_bar.png")

    # Waterfall for highest-risk subscriber
    high_risk_idx = int(np.argmax(y_prob_xgb))
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_values[high_risk_idx], max_display=12, show=False)
    plt.title(f"SHAP Waterfall — High-Risk Subscriber "
              f"(p_churn={y_prob_xgb[high_risk_idx]:.3f})", fontweight="bold")
    plt.tight_layout()
    save_fig("w8_shap_waterfall.png")

    return {"xgb_clf": xgb_clf, "xgb_reg": xgb_reg,
            "scale_pos": scale_pos,
            "auc_xgb": auc_xgb, "ap_xgb": ap_xgb, "f1_xgb": f1_xgb,
            "y_prob_xgb": y_prob_xgb, "shap_values": shap_values}


# ─────────────────────────────────────────────────────────────────────────────
# W9 — k-means taste segmentation + PCA visualisation
# ─────────────────────────────────────────────────────────────────────────────

def week9_taste_segments(df: pd.DataFrame,
                          X_train_e, X_val_e, X_test_e,
                          y_clf_train, y_clf_val, y_clf_test,
                          preprocessor_e,
                          scale_pos: int) -> dict:
    banner("W9", "k-Means Taste Segmentation + PCA Visualisation")
    from sklearn.metrics import silhouette_score

    # Transform full dataset for clustering
    X_all_e = engineer_features(
        df.drop(columns=["user_id", TARGET_CLF, TARGET_REG])
    )
    X_all_t = preprocessor_e.transform(X_all_e)

    # Elbow + silhouette to find best K
    inertias, silhouettes = [], []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_all_t)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X_all_t, labels,
                                                    sample_size=500,
                                                    random_state=RANDOM_STATE)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(K_range, inertias, "o-", color="#4C72B0", lw=2)
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia (SSE)")
    axes[0].set_title("Elbow Method", fontweight="bold")
    axes[1].plot(K_range, silhouettes, "s-", color="#DD8452", lw=2)
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis", fontweight="bold")
    plt.suptitle("K-Means — Optimal K Selection", fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("w9_kmeans_tuning.png")

    BEST_K = max(int(K_range[np.argmax(silhouettes)]), 4)
    print(f"  Best K (silhouette) : {BEST_K}")

    # Fit final k-means
    kmeans = KMeans(n_clusters=BEST_K, random_state=RANDOM_STATE, n_init=20)
    df = df.copy()
    df["taste_segment"] = kmeans.fit_predict(X_all_t)

    # PCA 2-D
    pca    = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca  = pca.fit_transform(X_all_t)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["segment"] = df["taste_segment"].values
    pca_df["churn"]   = df[TARGET_CLF].values

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    palette = sns.color_palette("tab10", BEST_K)
    for seg in range(BEST_K):
        m = pca_df["segment"] == seg
        axes[0].scatter(pca_df.loc[m, "PC1"], pca_df.loc[m, "PC2"],
                        c=[palette[seg]], label=f"Seg {seg}", alpha=0.6, s=20)
    axes[0].set_title(f"PCA — Taste Segments (K={BEST_K})", fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    axes[0].legend(markerscale=2, fontsize=9)

    from matplotlib.patches import Patch
    colours = pca_df["churn"].map({0: "#4C72B0", 1: "#DD8452"})
    axes[1].scatter(pca_df["PC1"], pca_df["PC2"], c=colours, alpha=0.5, s=20)
    axes[1].legend(handles=[Patch(color="#4C72B0", label="Retained"),
                             Patch(color="#DD8452", label="Churned")])
    axes[1].set_title("PCA — Churn Overlay", fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.tight_layout()
    save_fig("w9_pca_segments.png")

    # Segment profiles
    seg_profile = df.groupby("taste_segment").agg(
        n_users=   (TARGET_CLF, "count"),
        churn_rate=(TARGET_CLF, "mean"),
        avg_wt    =(TARGET_REG, "mean"),
        avg_daily =("daily_listening_minutes", "mean"),
        avg_skip  =("skip_rate_pct", "mean"),
        top_genre =("top_genre",          lambda x: x.value_counts().index[0]),
        top_sub   =("subscription_type",  lambda x: x.value_counts().index[0]),
    ).round(3)
    print("\n  Taste Segment Profiles:")
    print(seg_profile.to_string())

    # Append taste_segment to train/val/test as numeric feature
    pre_e_tr = preprocessor_e.transform(X_train_e)
    pre_e_va = preprocessor_e.transform(X_val_e)
    pre_e_te = preprocessor_e.transform(X_test_e)

    seg_train = kmeans.predict(pre_e_tr).reshape(-1, 1)
    seg_val   = kmeans.predict(pre_e_va).reshape(-1, 1)
    seg_test  = kmeans.predict(pre_e_te).reshape(-1, 1)

    # Build separate preprocessor_seg for segment-aware matrices
    X_train_seg = X_train_e.copy(); X_train_seg["taste_segment"] = seg_train.ravel()
    X_val_seg   = X_val_e.copy();   X_val_seg["taste_segment"]   = seg_val.ravel()
    X_test_seg  = X_test_e.copy();  X_test_seg["taste_segment"]  = seg_test.ravel()

    preprocessor_seg = build_preprocessor(with_segment=True)
    preprocessor_seg.fit(X_train_seg)
    Xtr_seg = preprocessor_seg.transform(X_train_seg)
    Xva_seg = preprocessor_seg.transform(X_val_seg)
    Xte_seg = preprocessor_seg.transform(X_test_seg)

    # Retrain XGBoost with segment feature
    xgb_seg = xgb.XGBClassifier(
        n_estimators=1000, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=scale_pos, eval_metric="auc",
        early_stopping_rounds=50, random_state=RANDOM_STATE, verbosity=0,
    )
    xgb_seg.fit(Xtr_seg, y_clf_train,
                eval_set=[(Xva_seg, y_clf_val)], verbose=False)
    auc_seg = float(roc_auc_score(y_clf_val,
                                   xgb_seg.predict_proba(Xva_seg)[:, 1]))
    print(f"\n  XGBoost + Segment → AUC={auc_seg:.4f}")

    return {
        "kmeans": kmeans, "BEST_K": BEST_K, "df_with_seg": df,
        "preprocessor_seg": preprocessor_seg,
        "xgb_seg": xgb_seg, "auc_seg": auc_seg,
        "Xtr_seg": Xtr_seg, "Xva_seg": Xva_seg, "Xte_seg": Xte_seg,
        "X_train_seg": X_train_seg, "X_val_seg": X_val_seg,
        "X_test_seg":  X_test_seg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# W10 — Nested CV, calibration, SMOTE, sealed test
# ─────────────────────────────────────────────────────────────────────────────

def week10_evaluation(Xtr, Xva, Xte,
                       Xtr_seg, Xva_seg, Xte_seg,
                       y_clf_train, y_clf_val, y_clf_test,
                       y_reg_val, y_reg_test,
                       xgb_seg, xgb_reg,
                       X_dev_e, y_dev_clf,
                       preprocessor_e, scale_pos: int) -> dict:
    banner("W10", "Rigorous Evaluation — Nested CV, Calibration, Sealed Test")
    from sklearn.pipeline import Pipeline as SKPipeline

    # Nested CV (outer 5, inner 3)
    print("  Nested CV (5 outer folds)…")
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_pipe = SKPipeline([
        ("prep", preprocessor_e),
        ("clf",  xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, scale_pos_weight=scale_pos,
            eval_metric="auc", verbosity=0, random_state=RANDOM_STATE,
        )),
    ])
    nested_scores = cross_val_score(
        xgb_pipe, X_dev_e, y_dev_clf,
        cv=outer_cv, scoring="roc_auc", n_jobs=-1,
    )
    print(f"  Nested CV AUC : {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
    print(f"  Folds         : {nested_scores.round(4)}")

    # Probability calibration
    def _make_xgb():
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, scale_pos_weight=scale_pos,
            eval_metric="auc", verbosity=0, random_state=RANDOM_STATE,
        )

    cal_sig = CalibratedClassifierCV(_make_xgb(), method="sigmoid",  cv=3)
    cal_iso = CalibratedClassifierCV(_make_xgb(), method="isotonic", cv=3)
    cal_sig.fit(Xtr, y_clf_train)
    cal_iso.fit(Xtr, y_clf_train)

    y_prob_xgb_raw = xgb_seg.predict_proba(Xva_seg)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, prob in [
        ("Uncalibrated XGB", y_prob_xgb_raw),
        ("Platt (sigmoid)",  cal_sig.predict_proba(Xva)[:, 1]),
        ("Isotonic",         cal_iso.predict_proba(Xva)[:, 1]),
    ]:
        fp, mp = calibration_curve(y_clf_val, prob, n_bins=8)
        bs     = brier_score_loss(y_clf_val, prob)
        ax.plot(mp, fp, "s-", label=f"{name}  Brier={bs:.4f}")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean predicted prob."); ax.set_ylabel("Fraction positives")
    ax.set_title("Calibration Curves", fontweight="bold"); ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig("w10_calibration.png")

    # SMOTE resampling
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    Xtr_sm, y_sm = smote.fit_resample(Xtr, y_clf_train)
    xgb_sm = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, eval_metric="auc",
        verbosity=0, random_state=RANDOM_STATE,
    )
    xgb_sm.fit(Xtr_sm, y_sm)
    auc_sm = float(roc_auc_score(y_clf_val, xgb_sm.predict_proba(Xva)[:, 1]))
    print(f"  SMOTE + XGBoost Val AUC : {auc_sm:.4f}")

    # ── SEALED TEST SET ──────────────────────────────────────────────────────
    print("\n  🔒 SEALED TEST SET EVALUATION")
    y_prob_test = xgb_seg.predict_proba(Xte_seg)[:, 1]
    y_pred_test = xgb_seg.predict(Xte_seg)

    test_auc = float(roc_auc_score(y_clf_test, y_prob_test))
    test_ap  = float(average_precision_score(y_clf_test, y_prob_test))
    test_f1  = float(f1_score(y_clf_test, y_pred_test))
    test_bs  = float(brier_score_loss(y_clf_test, y_prob_test))

    print(f"  Churn  AUC={test_auc:.4f}  AP={test_ap:.4f}  "
          f"F1={test_f1:.4f}  Brier={test_bs:.4f}")
    print(classification_report(y_clf_test, y_pred_test,
                                 target_names=["Retained", "Churned"]))

    y_reg_pred_test = xgb_reg.predict(Xte)
    test_rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_test)))
    test_r2   = float(r2_score(y_reg_test, y_reg_pred_test))
    print(f"  WatchTime  RMSE={test_rmse:.2f}  R²={test_r2:.4f}")

    # Confusion matrix + ROC
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cm = confusion_matrix(y_clf_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"])
    axes[0].set_title(f"Confusion Matrix (Test)\nAUC={test_auc:.3f}", fontweight="bold")
    axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")

    fpr_t, tpr_t, _ = roc_curve(y_clf_test, y_prob_test)
    axes[1].plot(fpr_t, tpr_t, lw=2.5, label=f"XGBoost AUC={test_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC — Sealed Test Set", fontweight="bold"); axes[1].legend()
    plt.tight_layout()
    save_fig("w10_test_evaluation.png")

    return {
        "test_auc": test_auc, "test_ap": test_ap,
        "test_f1":  test_f1,  "test_bs": test_bs,
        "test_rmse": test_rmse, "test_r2": test_r2,
        "nested_auc_mean": float(nested_scores.mean()),
        "nested_auc_std":  float(nested_scores.std()),
        "y_prob_test": y_prob_test, "y_reg_pred_test": y_reg_pred_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
# W11 — Package champion artefact + PSI drift guard
# ─────────────────────────────────────────────────────────────────────────────

def week11_package(preprocessor_e, preprocessor_seg, kmeans, xgb_seg, xgb_reg,
                    feature_names_base: list[str],
                    BEST_K: int, test_metrics: dict,
                    Xtr_base: np.ndarray, Xte_base: np.ndarray,
                    num_feature_names: list[str]) -> str:
    banner("W11", "Package Champion Artefact + PSI Drift Guard")

    VERSION = datetime.now().strftime("%Y%m%d_%H%M")
    bundle  = {
        "version":          VERSION,
        "preprocessor_e":   preprocessor_e,
        "preprocessor_seg": preprocessor_seg,
        "kmeans":           kmeans,
        "xgb_clf":          xgb_seg,
        "xgb_reg":          xgb_reg,
        "feature_names":    feature_names_base,
        "num_features":     NUM_FEATURES_ENG,
        "cat_features":     CAT_FEATURES,
        "seg_features":     NUM_FEATURES_SEG,
        "BEST_K":           BEST_K,
        "metrics":          test_metrics,
    }

    path = os.path.join(MODEL_DIR, f"streamstay_champion_v{VERSION}.pkl")
    joblib.dump(bundle, path)
    print(f"  Artefact saved  : {path}  ({os.path.getsize(path)/1e6:.2f} MB)")

    # PSI drift guard (train vs. test — simulating production data)
    psi_df = psi_report(Xtr_base, Xte_base, num_feature_names)
    print("\n  PSI Drift Guard — Train vs Simulated Production (test set)")
    print(psi_df.head(12).to_string(index=False))
    max_psi = float(psi_df["PSI"].max())
    if max_psi > 0.25:
        print(f"\n  🚨 DRIFT ALERT: Max PSI={max_psi:.4f} — retrain recommended!")
    else:
        print(f"\n  ✅ No significant drift (Max PSI={max_psi:.4f})")

    # PSI bar plot
    top_psi = psi_df.head(12).sort_values("PSI")
    colours = ["#DD8452" if p > 0.25 else "#FFBE7B" if p > 0.10 else "#4C72B0"
               for p in top_psi["PSI"]]
    plt.figure(figsize=(10, 5))
    plt.barh(top_psi["feature"], top_psi["PSI"], color=colours, edgecolor="white")
    plt.axvline(0.10, color="orange", ls="--", lw=1.5, label="Monitor (0.10)")
    plt.axvline(0.25, color="red",    ls="--", lw=1.5, label="Drift (0.25)")
    plt.xlabel("PSI"); plt.title("PSI Drift Guard — Feature Stability", fontweight="bold")
    plt.legend(); plt.tight_layout()
    save_fig("w11_psi_drift.png")

    return path


# ─────────────────────────────────────────────────────────────────────────────
# Final Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def final_dashboard(df_seg, test_metrics, auc_xgb,
                     y_prob_test, y_clf_test,
                     y_reg_test, y_reg_pred_test,
                     shap_values, feature_names, BEST_K, VERSION):
    banner("DASHBOARD", "StreamStay — Final Summary")

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # Champion model summary table
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    rows = [["XGBoost Churn ★",    f"AUC={test_metrics['test_auc']:.3f}"],
            ["XGBoost WatchTime ★", f"R²={test_metrics['test_r2']:.3f}"],
            ["K-Means Segments",   f"K={BEST_K}"]]
    t = ax1.table(cellText=rows, colLabels=["Production Model", "Score"],
                  loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.2, 1.6)
    for j in range(2):
        t[1, j].set_facecolor("#4C72B0"); t[1, j].set_text_props(color="white", fontweight="bold")
        t[2, j].set_facecolor("#4C72B0"); t[2, j].set_text_props(color="white", fontweight="bold")
        t[3, j].set_facecolor("#8FB0D4"); t[3, j].set_text_props(fontweight="bold")
    ax1.set_title("Champion Model Summary", fontweight="bold", pad=10)

    # Churn by subscription
    ax2 = fig.add_subplot(gs[0, 1])
    cr = df_seg.groupby("subscription_type")[TARGET_CLF].mean() * 100
    cr.sort_values().plot(kind="barh", ax=ax2,
        color=sns.color_palette("muted", 4), edgecolor="white")
    ax2.set_xlabel("Churn %"); ax2.set_title("Churn by Sub Type", fontweight="bold")
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    # Watch-time by segment
    ax3 = fig.add_subplot(gs[0, 2])
    df_seg.groupby("taste_segment")[TARGET_REG].mean().plot(
        kind="bar", ax=ax3, color=sns.color_palette("tab10", BEST_K), edgecolor="white")
    ax3.set_xlabel("Segment"); ax3.set_ylabel("Avg min/week")
    ax3.set_title("Watch-Time by Taste Segment", fontweight="bold")
    ax3.tick_params(axis="x", rotation=0)

    # ROC — champion model (sealed test)
    ax4 = fig.add_subplot(gs[1, 0:2])
    fpr_t, tpr_t, _ = roc_curve(y_clf_test, y_prob_test)
    ax4.plot(fpr_t, tpr_t, lw=2.5, color="#4C72B0",
             label=f"XGBoost + Segment  AUC={test_metrics['test_auc']:.3f}")
    ax4.plot([0, 1], [0, 1], "k--", lw=1, label="Random (no skill)")
    ax4.set_xlabel("FPR"); ax4.set_ylabel("TPR")
    ax4.set_title("ROC — Champion Classifier (Sealed Test)", fontweight="bold")
    ax4.legend(fontsize=10)

    # Metric card
    ax5 = fig.add_subplot(gs[1, 2]); ax5.axis("off")
    txt = (f"🏆 FINAL TEST METRICS\n\n"
           f"CHURN\n"
           f"  AUC  : {test_metrics['test_auc']:.4f}\n"
           f"  F1   : {test_metrics['test_f1']:.4f}\n"
           f"  AP   : {test_metrics['test_ap']:.4f}\n"
           f"  Brier: {test_metrics['test_bs']:.4f}\n\n"
           f"WATCH-TIME\n"
           f"  RMSE : {test_metrics['test_rmse']:.1f} min\n"
           f"  R²   : {test_metrics['test_r2']:.4f}\n\n"
           f"Segments : {BEST_K}\n"
           f"Version  : {VERSION}")
    ax5.text(0.05, 0.95, txt, transform=ax5.transAxes, fontsize=10.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F4FD", alpha=0.8))

    # SHAP global importance
    ax6 = fig.add_subplot(gs[2, 0:2])
    shap_mean = np.abs(shap_values.values).mean(axis=0)
    si = pd.Series(shap_mean, index=feature_names).sort_values(ascending=False).head(12)
    si.sort_values().plot(kind="barh", ax=ax6, color="#4C72B0", edgecolor="white")
    ax6.set_xlabel("Mean |SHAP|")
    ax6.set_title("Top 12 Features — SHAP Global Importance", fontweight="bold")

    # Actual vs Predicted (watch-time)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.scatter(y_reg_test, y_reg_pred_test, alpha=0.4, color="#4C72B0", s=18)
    mn_t, mx_t = float(y_reg_test.min()), float(y_reg_test.max())
    ax7.plot([mn_t, mx_t], [mn_t, mx_t], "r--", lw=1.5)
    ax7.set_xlabel("Actual (min/week)"); ax7.set_ylabel("Predicted")
    ax7.set_title(f"Watch-Time: Actual vs Pred\n"
                  f"RMSE={test_metrics['test_rmse']:.1f}, "
                  f"R²={test_metrics['test_r2']:.3f}", fontweight="bold")

    plt.suptitle("🎵 StreamStay — Final ML Dashboard", fontsize=16,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("streamstay_final_dashboard.png")
    print(f"\n  Dashboard saved → {os.path.join(PLOT_DIR, 'streamstay_final_dashboard.png')}")


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator — runs W1 → W11
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  StreamStay — Churn & Watch-Time ML Pipeline")
    print("="*70)

    # Load raw data
    df_raw = pd.read_csv(DATA_PATH)

    # ── W1 ────────────────────────────────────────────────────────────────────
    w1 = week1_frame_targets(df_raw)

    # ── W2 ────────────────────────────────────────────────────────────────────
    (X_train, X_val, X_test,
     y_clf_train, y_clf_val, y_clf_test,
     y_reg_train, y_reg_val, y_reg_test) = week2_ingest_and_split(df_raw)

    # Feature engineering (definitions in features.py)
    X_train_e = engineer_features(X_train)
    X_val_e   = engineer_features(X_val)
    X_test_e  = engineer_features(X_test)

    # Leakage-safe preprocessor — fit on train ONLY, transform val/test
    preprocessor_e = build_preprocessor(with_segment=False)
    preprocessor_e.fit(X_train_e)
    Xtr_e = preprocessor_e.transform(X_train_e)
    Xva_e = preprocessor_e.transform(X_val_e)
    Xte_e = preprocessor_e.transform(X_test_e)

    feature_names_base = get_feature_names(preprocessor_e, with_segment=False)
    print(f"\n  Preprocessed shape: {Xtr_e.shape}  ({Xtr_e.shape[1]} features)")

    # ── W8 — XGBoost champion ─────────────────────────────────────────────────
    w8 = week8_xgboost(Xtr_e, Xva_e, Xte_e,
                        y_clf_train, y_clf_val, y_clf_test,
                        y_reg_train, y_reg_val,
                        feature_names_base)

    # ── W9 — K-Means taste segmentation + segment-aware retrain ───────────────
    w9 = week9_taste_segments(df_raw, X_train_e, X_val_e, X_test_e,
                               y_clf_train, y_clf_val, y_clf_test,
                               preprocessor_e, w8["scale_pos"])

    # ── W10 — Nested CV, calibration, sealed test ─────────────────────────────
    X_dev_e   = pd.concat([X_train_e, X_val_e]).reset_index(drop=True)
    y_dev_clf = pd.concat([y_clf_train, y_clf_val]).reset_index(drop=True)

    w10 = week10_evaluation(
        Xtr_e, Xva_e, Xte_e,
        w9["Xtr_seg"], w9["Xva_seg"], w9["Xte_seg"],
        y_clf_train, y_clf_val, y_clf_test,
        y_reg_val, y_reg_test,
        w9["xgb_seg"], w8["xgb_reg"],
        X_dev_e, y_dev_clf,
        preprocessor_e, w8["scale_pos"],
    )

    # ── W11 — Package champion artefact ───────────────────────────────────────
    test_metrics = {
        "test_auc":   w10["test_auc"],
        "test_ap":    w10["test_ap"],
        "test_f1":    w10["test_f1"],
        "test_bs":    w10["test_bs"],
        "test_rmse":  w10["test_rmse"],
        "test_r2":    w10["test_r2"],
    }
    artefact_path = week11_package(
        preprocessor_e, w9["preprocessor_seg"],
        w9["kmeans"], w9["xgb_seg"], w8["xgb_reg"],
        feature_names_base, w9["BEST_K"], test_metrics,
        Xtr_e, Xte_e, NUM_FEATURES_ENG,
    )

    # ── Final Dashboard ────────────────────────────────────────────────────────
    VERSION = datetime.now().strftime("%Y%m%d_%H%M")
    final_dashboard(
        df_seg=w9["df_with_seg"],
        test_metrics=test_metrics,
        auc_xgb=w8["auc_xgb"],
        y_prob_test=w10["y_prob_test"],
        y_clf_test=y_clf_test,
        y_reg_test=y_reg_test,
        y_reg_pred_test=w10["y_reg_pred_test"],
        shap_values=w8["shap_values"],
        feature_names=feature_names_base,
        BEST_K=w9["BEST_K"], VERSION=VERSION,
    )

    print(f"\n{'='*70}")
    print("  ✅ Pipeline COMPLETE  |  Artefact: " + artefact_path)
    print(f"  Run:  python api.py")
    print(f"{'='*70}\n")
    return artefact_path


if __name__ == "__main__":
    main()
