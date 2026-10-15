"""
create_presentation.py — StreamStay ML Presentation Generator (Clean & Minimalist)
===================================================================================
A clean, visual, and concise 16:9 widescreen presentation:
1. Title Slide (with dedicated space for teammate names)
2. Introduction & Problem Statement
3. How We Are Solving It (The 3-Step Pipeline)
4. Model 1: XGBoost Churn Classifier (Why used, results + SHAP plot)
5. Model 2: XGBoost Watch-Time Regressor (Why used, results + feature plot)
6. Model 3: K-Means Taste Segmentation (Why used, archetypes + tuning plot)
7. Model 4: PCA Dimensionality Reduction (Verification + 2D scatter plot)
8. Why Baselines Were Retired (Short, clear 1-line rationales)
9. Production Quality: Calibration & PSI Drift Guard (with drift plot)
10. Project Deployment & Live Links (Clean, spacious link placeholders)
11. Thank You & Q&A Slide
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE


def build_streamstay_presentation(output_path="StreamStay Project Overview.pptx"):
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # ── Color Palette ────────────────────────────────────────────────────────
    DARK_BG = RGBColor(15, 23, 42)             # Slate 900
    DARK_CARD = RGBColor(30, 41, 59)           # Slate 800
    DARK_BORDER = RGBColor(51, 65, 85)         # Slate 700
    DARK_TITLE = RGBColor(248, 250, 252)       # White / Slate 50
    DARK_SUB = RGBColor(148, 163, 184)         # Slate 400
    DARK_ACCENT = RGBColor(129, 140, 248)      # Indigo 400

    BG_COLOR = RGBColor(248, 250, 252)         # Crisp Slate 50
    CARD_BG = RGBColor(255, 255, 255)          # Pure White
    CARD_BORDER = RGBColor(226, 232, 240)      # Slate 200
    TEXT_MAIN = RGBColor(15, 23, 42)           # Slate 900
    TEXT_MUTED = RGBColor(100, 116, 139)       # Slate 500
    TEXT_BODY = RGBColor(51, 65, 85)           # Slate 700

    ACCENT_INDIGO = RGBColor(79, 70, 229)      # Indigo 600
    ACCENT_TEAL = RGBColor(13, 148, 136)       # Teal 600
    ACCENT_AMBER = RGBColor(217, 119, 6)       # Amber 600
    ACCENT_ROSE = RGBColor(225, 29, 72)        # Rose 600
    ACCENT_GREEN = RGBColor(22, 163, 74)       # Green 600
    CARD_HIGHLIGHT = RGBColor(238, 242, 255)   # Light Indigo tint

    # ── Helpers ──────────────────────────────────────────────────────────────

    def set_bg(slide, color=BG_COLOR):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_header(slide, tag, title, subtitle=""):
        # Category Tag
        t_box = slide.shapes.add_textbox(Inches(0.9), Inches(0.55), Inches(11.5), Inches(0.35))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = tag.upper()
        p_t.font.size = Pt(10)
        p_t.font.bold = True
        p_t.font.color.rgb = ACCENT_INDIGO

        # Main Title
        m_box = slide.shapes.add_textbox(Inches(0.9), Inches(0.9), Inches(11.5), Inches(0.6))
        tf_m = m_box.text_frame
        tf_m.word_wrap = True
        tf_m.margin_left = tf_m.margin_right = tf_m.margin_top = tf_m.margin_bottom = 0
        p_m = tf_m.paragraphs[0]
        p_m.text = title
        p_m.font.size = Pt(23)
        p_m.font.bold = True
        p_m.font.color.rgb = TEXT_MAIN

        # Subtitle
        if subtitle:
            s_box = slide.shapes.add_textbox(Inches(0.9), Inches(1.5), Inches(11.5), Inches(0.35))
            tf_s = s_box.text_frame
            tf_s.word_wrap = True
            tf_s.margin_left = tf_s.margin_right = tf_s.margin_top = tf_s.margin_bottom = 0
            p_s = tf_s.paragraphs[0]
            p_s.text = subtitle
            p_s.font.size = Pt(12)
            p_s.font.color.rgb = TEXT_MUTED

    def add_clean_card(slide, left, top, width, height, title, points, badge="", badge_color=ACCENT_INDIGO, top_color=None):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1)

        if top_color:
            bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left + Inches(0.05), top, width - Inches(0.1), Inches(0.06))
            bar.fill.solid()
            bar.fill.fore_color.rgb = top_color
            bar.line.fill.background()

        tb = slide.shapes.add_textbox(left + Inches(0.3), top + Inches(0.3), width - Inches(0.6), height - Inches(0.6))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

        first_p = tf.paragraphs[0]
        if badge:
            first_p.text = badge.upper()
            first_p.font.size = Pt(9)
            first_p.font.bold = True
            first_p.font.color.rgb = badge_color
            first_p.space_after = Pt(4)
            p_title = tf.add_paragraph()
        else:
            p_title = first_p

        p_title.text = title
        p_title.font.size = Pt(14)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_MAIN
        p_title.space_after = Pt(14)

        for pt in points:
            p = tf.add_paragraph()
            p.space_after = Pt(8)
            if isinstance(pt, tuple):
                lead, text = pt
                r_l = p.add_run()
                r_l.text = f"- {lead}: "
                r_l.font.bold = True
                r_l.font.size = Pt(11)
                r_l.font.color.rgb = TEXT_MAIN

                r_t = p.add_run()
                r_t.text = text
                r_t.font.size = Pt(11)
                r_t.font.color.rgb = TEXT_BODY
            else:
                r = p.add_run()
                r.text = f"- {pt}"
                r.font.size = Pt(11)
                r.font.color.rgb = TEXT_BODY

    def add_stat_box(slide, left, top, width, height, value, label, sub=""):
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        box.fill.solid()
        box.fill.fore_color.rgb = CARD_BG
        box.line.color.rgb = CARD_BORDER
        box.line.width = Pt(1)

        tb = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.12), width - Inches(0.3), height - Inches(0.24))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

        p_v = tf.paragraphs[0]
        p_v.text = str(value)
        p_v.font.size = Pt(18)
        p_v.font.bold = True
        p_v.font.color.rgb = ACCENT_INDIGO

        p_l = tf.add_paragraph()
        p_l.text = label
        p_l.font.size = Pt(9.5)
        p_l.font.bold = True
        p_l.font.color.rgb = TEXT_MAIN

        if sub:
            p_s = tf.add_paragraph()
            p_s.text = sub
            p_s.font.size = Pt(8.5)
            p_s.font.color.rgb = TEXT_MUTED

    def add_plot(slide, left, top, width, height, filename, caption=""):
        path = os.path.join("plots", filename)
        if not os.path.exists(path):
            path = filename

        frame = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        frame.fill.solid()
        frame.fill.fore_color.rgb = RGBColor(255, 255, 255)
        frame.line.color.rgb = CARD_BORDER
        frame.line.width = Pt(1)

        if os.path.exists(path):
            pad = Inches(0.1)
            cap_h = Inches(0.3) if caption else Inches(0)
            slide.shapes.add_picture(path, left + pad, top + pad, width - (pad * 2), height - (pad * 2) - cap_h)

            if caption:
                tb = slide.shapes.add_textbox(left + pad, top + height - cap_h, width - (pad * 2), cap_h)
                tf = tb.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.text = caption
                p.font.size = Pt(9)
                p.font.color.rgb = TEXT_MUTED
                p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 1: Title Slide (Simple, Clean & Teammate Space)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_bg(s1, DARK_BG)

    # Title & Subtitle
    t_box = s1.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(10.9), Inches(2.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True

    p_badge = tf1.paragraphs[0]
    p_badge.text = "MACHINE LEARNING PROJECT"
    p_badge.font.size = Pt(11)
    p_badge.font.bold = True
    p_badge.font.color.rgb = DARK_ACCENT
    p_badge.space_after = Pt(8)

    p_title = tf1.add_paragraph()
    p_title.text = "StreamStay: Churn & Watch-Time Analytics"
    p_title.font.size = Pt(36)
    p_title.font.bold = True
    p_title.font.color.rgb = DARK_TITLE
    p_title.space_after = Pt(10)

    p_sub = tf1.add_paragraph()
    p_sub.text = "Predicting 30-day subscriber churn and weekly listening engagement using segment-aware machine learning."
    p_sub.font.size = Pt(15)
    p_sub.font.color.rgb = DARK_SUB

    # Teammates Box
    team_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(4.2), Inches(10.9), Inches(2.3))
    team_card.fill.solid()
    team_card.fill.fore_color.rgb = DARK_CARD
    team_card.line.color.rgb = DARK_BORDER
    team_card.line.width = Pt(1)

    team_tb = s1.shapes.add_textbox(Inches(1.5), Inches(4.45), Inches(10.3), Inches(1.8))
    tf_team = team_tb.text_frame
    tf_team.word_wrap = True

    p_th = tf_team.paragraphs[0]
    p_th.text = "TEAM MEMBERS"
    p_th.font.size = Pt(10)
    p_th.font.bold = True
    p_th.font.color.rgb = DARK_ACCENT
    p_th.space_after = Pt(12)

    team_slots = [
        "Member 1: [ Name ] — Roll No: [ Enter Roll No ]",
        "Member 2: [ Name ] — Roll No: [ Enter Roll No ]",
        "Member 3: [ Name ] — Roll No: [ Enter Roll No ]",
        "Member 4: [ Name ] — Roll No: [ Enter Roll No ]",
    ]
    for slot in team_slots:
        p_m = tf_team.add_paragraph()
        p_m.space_after = Pt(5)
        r = p_m.add_run()
        r.text = f"- {slot}"
        r.font.size = Pt(11)
        r.font.color.rgb = DARK_TITLE

    # =========================================================================
    # SLIDE 2: Introduction & Problem Statement
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_bg(s2)
    add_header(s2, "Introduction & Problem Statement",
               "The Challenge: Silent Churn in Music Streaming",
               "Why subscriber retention matters more than acquisition for streaming platforms.")

    col_w = Inches(3.64)
    top_p = Inches(2.1)
    card_h = Inches(4.7)

    add_clean_card(s2, Inches(0.9), top_p, col_w, card_h,
                   "The Streaming Industry",
                   [
                       ("Zero Switching Cost", "All major music apps share the same 100M+ song catalog."),
                       ("One-Click Exit", "Users can cancel their plan in seconds from their phone."),
                       ("Acquisition is Expensive", "Acquiring a new user costs 5x-7x more than keeping an existing one.")
                   ],
                   badge="Context", badge_color=ACCENT_AMBER, top_color=ACCENT_AMBER)

    add_clean_card(s2, Inches(4.84), top_p, col_w, card_h,
                   "The 'Silent Churn' Problem",
                   [
                       ("Gradual Decay", "Users do not cancel abruptly; their activity fades over several weeks."),
                       ("Early Symptoms", "Rising song skip rate, fewer app opens, and declining daily minutes."),
                       ("Missed Intervention", "Platforms usually notice only after the cancellation button is pressed.")
                   ],
                   badge="The Problem", badge_color=ACCENT_ROSE, top_color=ACCENT_ROSE)

    add_clean_card(s2, Inches(8.78), top_p, col_w, card_h,
                   "What StreamStay Solves",
                   [
                       ("30-Day Churn Risk", "Predict who is likely to churn before they make the decision."),
                       ("Watch-Time Forecast", "Estimate expected weekly listening minutes to measure value."),
                       ("Actionable CRM", "Trigger automated retention perks (discounts, playlists) to save at-risk users.")
                   ],
                   badge="Our Objective", badge_color=ACCENT_TEAL, top_color=ACCENT_TEAL)

    # =========================================================================
    # SLIDE 3: How Are We Solving It (Pipeline Overview)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_bg(s3)
    add_header(s3, "System Architecture",
               "How We Solve It: The 3-Step Machine Learning Pipeline",
               "From raw subscriber behavior to segment-aware predictions.")

    step_w = Inches(3.64)

    add_clean_card(s3, Inches(0.9), top_p, step_w, card_h,
                   "1. Ingestion & Ratios",
                   [
                       ("1,200 Subscribers", "Combines listening habits, device, plan type, and billing history."),
                       ("Leakage-Safe Split", "Strict 60/20/20 train/validation/test split."),
                       ("Engineered Features", "Created daily song density, skip ratios, and inactivity velocity.")
                   ],
                   badge="Step 1: Data", badge_color=ACCENT_INDIGO, top_color=ACCENT_INDIGO)

    add_clean_card(s3, Inches(4.84), top_p, step_w, card_h,
                   "2. Taste Segmentation",
                   [
                       ("K-Means Clustering", "Discovers latent listener personas (e.g. casual vs. power users)."),
                       ("Closed-Loop Feedback", "The discovered segment is fed directly into the churn model."),
                       ("Persona-Specific Risk", "Enables customized churn decision boundaries per archetype.")
                   ],
                   badge="Step 2: Unsupervised", badge_color=ACCENT_TEAL, top_color=ACCENT_TEAL)

    add_clean_card(s3, Inches(8.78), top_p, step_w, card_h,
                   "3. Dual XGBoost Scoring",
                   [
                       ("Churn Classifier", "Outputs calibrated churn probability (0-100%) and risk tier."),
                       ("Watch Regressor", "Forecasts expected weekly minutes (min/week) with 98% accuracy."),
                       ("FastAPI Deployment", "Production endpoints (/predict, /drift) ready for live integration.")
                   ],
                   badge="Step 3: Supervised", badge_color=ACCENT_GREEN, top_color=ACCENT_GREEN)

    # =========================================================================
    # SLIDE 4: Model 1 — XGBoost Churn Classifier
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_bg(s4)
    add_header(s4, "Supervised Model 1",
               "XGBoost Classifier: 30-Day Churn Prediction",
               "Predicting whether a subscriber will cancel within the next 30 days.")

    left_w = Inches(5.5)
    right_w = Inches(5.9)

    add_clean_card(s4, Inches(0.9), top_p, left_w, Inches(3.3),
                   "Why XGBoost & Key Findings",
                   [
                       ("Handles Class Imbalance", "Uses scale_pos_weight to catch rare churners without noisy synthetic data."),
                       ("Captures Thresholds", "Identifies that inactivity past 20 days combined with skips triggers churn."),
                       ("Segment-Aware", "Adjusts risk thresholds based on the user's K-Means taste cluster."),
                       ("Calibrated Probabilities", "Platt scaling ensures a 70% predicted score reflects 70% true churn risk.")
                   ],
                   badge="Model Rationale", badge_color=ACCENT_INDIGO, top_color=ACCENT_INDIGO)

    # Metric pills
    add_stat_box(s4, Inches(0.9), Inches(5.55), Inches(1.7), Inches(1.25), "0.7042", "ROC-AUC", "Sealed Test")
    add_stat_box(s4, Inches(2.8), Inches(5.55), Inches(1.7), Inches(1.25), "0.2815", "Avg Precision", "PR-AUC")
    add_stat_box(s4, Inches(4.7), Inches(5.55), Inches(1.7), Inches(1.25), "0.1782", "Brier Score", "Calibrated")

    # Plot
    add_plot(s4, Inches(6.7), top_p, right_w, Inches(4.7),
             "w8_shap_summary.png",
             "SHAP Summary: Inactivity days, skip rate & payment failures are top churn drivers")

    # =========================================================================
    # SLIDE 5: Model 2 — XGBoost Watch-Time Regressor
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_bg(s5)
    add_header(s5, "Supervised Model 2",
               "XGBoost Regressor: Weekly Watch-Time Forecast",
               "Predicting expected weekly listening minutes to measure customer engagement.")

    add_clean_card(s5, Inches(0.9), top_p, left_w, Inches(3.3),
                   "Why XGBoost Regressor & Key Findings",
                   [
                       ("Outlier Robustness", "Standard linear models broke on extreme listeners (>500 min). XGBoost handles them cleanly."),
                       ("Non-Linear Dynamics", "Models diminishing returns between app opens and total streamed minutes."),
                       ("Feature Synergies", "Captures cross-interactions like daily songs multiplied by offline listening."),
                       ("Actionable Tiers", "Accurately categorizes users into LOW, MEDIUM, and HIGH engagement bands.")
                   ],
                   badge="Model Rationale", badge_color=ACCENT_AMBER, top_color=ACCENT_AMBER)

    add_stat_box(s5, Inches(0.9), Inches(5.55), Inches(2.6), Inches(1.25), "0.9812 R²", "Variance Explained", ">98% variance captured")
    add_stat_box(s5, Inches(3.8), Inches(5.55), Inches(2.6), Inches(1.25), "59.4 min", "RMSE Deviation", "Minimal weekly error")

    add_plot(s5, Inches(6.7), top_p, right_w, Inches(4.7),
             "w8_shap_bar.png",
             "Feature Importance: Daily listening habits and session frequency dominate watch-time")

    # =========================================================================
    # SLIDE 6: Model 3 — K-Means Taste Segmentation
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_bg(s6)
    add_header(s6, "Unsupervised Model 1",
               "K-Means: Listener Taste Segmentation",
               "Grouping subscribers into behavioral archetypes that feed into churn scoring.")

    add_clean_card(s6, Inches(0.9), top_p, left_w, Inches(3.3),
                   "Why K-Means & Discovered Archetypes",
                   [
                       ("100% Coverage", "Guarantees a clean discrete cluster for every subscriber without discarding noise."),
                       ("Tuned with Silhouette", "Swept K=2 to 10; K=4 yielded the clearest separation and lowest inertia."),
                       ("Discovered Personas", "1. Casual Mobile Free, 2. Power Audiophiles, 3. Commute Streamers, 4. Lapsing."),
                       ("The Closed Loop", "Segment ID is passed to XGBoost so churn risk is evaluated per persona.")
                   ],
                   badge="Unsupervised Grouping", badge_color=ACCENT_TEAL, top_color=ACCENT_TEAL)

    add_stat_box(s6, Inches(0.9), Inches(5.55), Inches(2.6), Inches(1.25), "K = 4", "Optimal Archetypes", "Selected via Elbow Method")
    add_stat_box(s6, Inches(3.8), Inches(5.55), Inches(2.6), Inches(1.25), "0.421", "Silhouette Score", "High cluster cohesion")

    add_plot(s6, Inches(6.7), top_p, right_w, Inches(4.7),
             "w9_kmeans_tuning.png",
             "K-Means Tuning: Elbow Method & Silhouette Score confirming K=4 optimal clusters")

    # =========================================================================
    # SLIDE 7: Model 4 — PCA Dimensionality Reduction
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_bg(s7)
    add_header(s7, "Unsupervised Model 2",
               "PCA: 2D Cluster Verification & Diagnostics",
               "Verifying that listener segments are genuinely separated in latent space.")

    add_clean_card(s7, Inches(0.9), top_p, left_w, Inches(3.3),
                   "Why PCA & What It Confirms",
                   [
                       ("Visual Sanity Check", "Reduces 28 feature dimensions to 2 to visually inspect cluster health."),
                       ("PC1 (Volume)", "Captures total listening volume (casual listeners on left, heavy users on right)."),
                       ("PC2 (Exploration)", "Captures taste breadth (single-artist repeaters vs. eclectic explorers)."),
                       ("Confirmed Separation", "Proves that subscriber personas form natural, distinct clusters.")
                   ],
                   badge="Visual Verification", badge_color=ACCENT_INDIGO, top_color=ACCENT_INDIGO)

    add_stat_box(s7, Inches(0.9), Inches(5.55), Inches(2.6), Inches(1.25), "28 Dims -> 2", "Orthogonal Projection", "Visualizes full space")
    add_stat_box(s7, Inches(3.8), Inches(5.55), Inches(2.6), Inches(1.25), "No Overlap", "Clear Separability", "Verifies K-Means quality")

    add_plot(s7, Inches(6.7), top_p, right_w, Inches(4.7),
             "w9_pca_segments.png",
             "PCA 2D Scatter: Clear visual boundary between subscriber segments")

    # =========================================================================
    # SLIDE 8: Why Baselines Were Retired
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_bg(s8)
    add_header(s8, "Model Selection",
               "Why Other Models Were Retired",
               "Eliminating redundant baselines to build a clean, high-performance production pipeline.")

    full_w = Inches(11.53)
    card_retired = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), top_p, full_w, Inches(4.7))
    card_retired.fill.solid()
    card_retired.fill.fore_color.rgb = CARD_BG
    card_retired.line.color.rgb = CARD_BORDER
    card_retired.line.width = Pt(1)

    t_box = s8.shapes.add_textbox(Inches(1.2), Inches(2.3), Inches(10.9), Inches(4.3))
    tf_ret = t_box.text_frame
    tf_ret.word_wrap = True

    p_title = tf_ret.paragraphs[0]
    p_title.text = "RETIRED BASELINES & RATIONALE"
    p_title.font.size = Pt(10)
    p_title.font.bold = True
    p_title.font.color.rgb = ACCENT_INDIGO
    p_title.space_after = Pt(14)

    retired_list = [
        ("OLS / SGD Regressor", "Rigid linear assumption; broke on heavy listeners with massive residual errors. Replaced by XGBoost Regressor."),
        ("Ridge & Lasso Sweep", "Linear L1/L2 penalty search was subsumed by XGBoost's built-in reg_alpha and depth pruning."),
        ("Logistic Regression", "Linear decision boundaries cannot capture multi-condition churn rules. Yielded poor ROC-AUC (0.61)."),
        ("Single Decision Tree", "High variance and severe overfitting on training data. Brittle rules that failed on test splits."),
        ("Random Forest", "Strong baseline, but XGBoost achieved higher precision-recall on imbalanced data with faster inference.")
    ]

    for model, reason in retired_list:
        p = tf_ret.add_paragraph()
        p.space_after = Pt(10)
        r_m = p.add_run()
        r_m.text = f"- {model}: "
        r_m.font.bold = True
        r_m.font.size = Pt(11)
        r_m.font.color.rgb = TEXT_MAIN

        r_r = p.add_run()
        r_r.text = reason
        r_r.font.size = Pt(11)
        r_r.font.color.rgb = TEXT_BODY

    p_v = tf_ret.add_paragraph()
    p_v.space_before = Pt(8)
    r_v = p_v.add_run()
    r_v.text = "Result: "
    r_v.font.bold = True
    r_v.font.size = Pt(11.5)
    r_v.font.color.rgb = ACCENT_GREEN

    r_vt = p_v.add_run()
    r_vt.text = "A focused 3-model system (XGBoost Classifier + XGBoost Regressor + K-Means) that is easier to deploy, faster to run, and more accurate."
    r_vt.font.size = Pt(11)
    r_vt.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # SLIDE 9: Production Quality: Calibration & Drift Guard
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_bg(s9)
    add_header(s9, "Production Reliability",
               "Model Reliability: Calibration & PSI Drift Guard",
               "Ensuring predictions remain trustworthy and monitoring data drift in real-time.")

    add_clean_card(s9, Inches(0.9), top_p, left_w, Inches(3.3),
                   "MLOps Quality Controls",
                   [
                       ("Platt Probability Calibration", "Raw tree scores can be overconfident. Calibration aligns predicted scores with true historical churn."),
                       ("PSI Data Drift Guard", "Continuously tracks distribution stability across 12 numeric features."),
                       ("Drift Thresholds", "PSI < 0.10 indicates healthy data. PSI >= 0.20 triggers an automated retrain alert."),
                       ("Current Health", "All 12 production features are stable (Max PSI = 0.041), requiring no retraining.")
                   ],
                   badge="MLOps Guardrails", badge_color=ACCENT_GREEN, top_color=ACCENT_GREEN)

    add_stat_box(s9, Inches(0.9), Inches(5.55), Inches(1.7), Inches(1.25), "0.0412", "Max PSI", "Well below 0.10 limit")
    add_stat_box(s9, Inches(2.8), Inches(5.55), Inches(1.7), Inches(1.25), "0 Drift", "Feature Alerts", "12/12 features stable")
    add_stat_box(s9, Inches(4.7), Inches(5.55), Inches(1.7), Inches(1.25), "0.178", "Brier Score", "Calibrated confidence")

    add_plot(s9, Inches(6.7), top_p, right_w, Inches(4.7),
             "w11_psi_drift.png",
             "PSI Drift Guard: All 12 telemetry features confirmed stable in safe zone")

    # =========================================================================
    # SLIDE 10: Project Deployment & Live Links
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_bg(s10)
    add_header(s10, "System Access",
               "Project Deployment & Live Access Links",
               "Full-stack interactive application deployed and accessible for evaluation.")

    dep_w = Inches(5.6)

    # Left: Stack Summary
    add_clean_card(s10, Inches(0.9), top_p, dep_w, Inches(4.7),
                   "The Deployed System",
                   [
                       ("FastAPI Backend", "Asynchronous Python API serving /predict, /predict/batch, /health, and /drift endpoints."),
                       ("Interactive React UI", "Vite dashboard with live subscriber testing, risk badges, and CRM retention suggestions."),
                       ("Packaged Model Bundle", "Single champion artefact (.pkl) containing preprocessors, K-Means, and XGBoost models."),
                       ("Docker Ready", "Containerized setup allowing one-command deployment to any cloud provider.")
                   ],
                   badge="Architecture", badge_color=ACCENT_INDIGO, top_color=ACCENT_INDIGO)

    # Right: Clean, Prominent Space for Links
    links_card = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), top_p, dep_w, Inches(4.7))
    links_card.fill.solid()
    links_card.fill.fore_color.rgb = CARD_HIGHLIGHT
    links_card.line.color.rgb = ACCENT_INDIGO
    links_card.line.width = Pt(1.5)

    l_tb = s10.shapes.add_textbox(Inches(7.1), Inches(2.3), Inches(5.0), Inches(4.3))
    tf_l = l_tb.text_frame
    tf_l.word_wrap = True

    p_lh = tf_l.paragraphs[0]
    p_lh.text = "DEPLOYMENT & ACCESS LINKS"
    p_lh.font.size = Pt(10)
    p_lh.font.bold = True
    p_lh.font.color.rgb = ACCENT_INDIGO
    p_lh.space_after = Pt(16)

    links_items = [
        ("Live Web Application", "[ Paste Live Dashboard Link Here ]\ne.g., http://localhost:5173 or deployed URL"),
        ("FastAPI Swagger Docs", "[ Paste API Documentation Link Here ]\ne.g., http://localhost:8000/docs"),
        ("GitHub Repository", "[ Paste GitHub Project Repository Link Here ]\ne.g., https://github.com/your-username/StreamStay-ML"),
        ("Docker Hub Container", "[ Paste Docker Hub Image Link Here ]\ne.g., docker pull username/streamstay-api:latest")
    ]

    for label, link in links_items:
        p = tf_l.add_paragraph()
        p.space_after = Pt(10)

        r_lbl = p.add_run()
        r_lbl.text = f"{label}:\n"
        r_lbl.font.bold = True
        r_lbl.font.size = Pt(11)
        r_lbl.font.color.rgb = TEXT_MAIN

        r_lnk = p.add_run()
        r_lnk.text = link
        r_lnk.font.size = Pt(9.5)
        r_lnk.font.color.rgb = ACCENT_INDIGO

    # =========================================================================
    # SLIDE 11: Thank You Slide (Clean & Modern Dark)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_bg(s11, DARK_BG)

    # Thank you box
    ty_box = s11.shapes.add_textbox(Inches(1.5), Inches(2.2), Inches(10.3), Inches(3.2))
    tf_ty = ty_box.text_frame
    tf_ty.word_wrap = True

    p_badge = tf_ty.paragraphs[0]
    p_badge.text = "STREAMSTAY ML CAPSTONE"
    p_badge.font.size = Pt(11)
    p_badge.font.bold = True
    p_badge.font.color.rgb = DARK_ACCENT
    p_badge.space_after = Pt(12)

    p_ty = tf_ty.add_paragraph()
    p_ty.text = "Thank You!"
    p_ty.font.size = Pt(44)
    p_ty.font.bold = True
    p_ty.font.color.rgb = DARK_TITLE
    p_ty.space_after = Pt(12)

    p_sub = tf_ty.add_paragraph()
    p_sub.text = "Questions & Discussion"
    p_sub.font.size = Pt(20)
    p_sub.font.color.rgb = DARK_SUB
    p_sub.space_after = Pt(24)

    p_note = tf_ty.add_paragraph()
    p_note.text = "StreamStay: Transforming subscriber churn from an unavoidable loss into an automated, data-driven retention engine."
    p_note.font.size = Pt(12)
    p_note.font.italic = True
    p_note.font.color.rgb = RGBColor(148, 163, 184)

    # Save presentation (with fallback if PowerPoint is locking the target file)
    try:
        prs.save(output_path)
        final_path = output_path
    except PermissionError:
        fallback_path = "StreamStay_Presentation.pptx"
        prs.save(fallback_path)
        final_path = fallback_path
        print(f"Notice: '{output_path}' is currently open in PowerPoint and locked. Saved clean version to: '{fallback_path}'.")

    print(f"Clean, streamlined presentation successfully generated at: {final_path}")
    return final_path


if __name__ == "__main__":
    build_streamstay_presentation("StreamStay Project Overview.pptx")
