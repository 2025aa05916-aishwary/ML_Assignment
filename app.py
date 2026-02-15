# app.py
import os
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from model.train_and_evaluate import (
    run_experiment, load_dataset, evaluate_model,
    make_test_csv, save_model_pkl, pick_best_model, ensure_dirs
)

# ---------- Page config ----------
st.set_page_config(page_title="ML Assignment 2 – Classification Bench", layout="wide")
st.title("ML Assignment 2 – End‑to‑End Classification App")
st.markdown(
    "This app trains **six classifiers** on the Breast Cancer Wisconsin dataset, "
    "compares metrics, lets you **upload a test CSV**, and **download** a sample test CSV "
    "and the **best model (.pkl)**."
)

# ---------- Helpers ----------
def _read_bytes(path: str) -> Optional[bytes]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

def _format_classification_report(report_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Turn sklearn classification_report(output_dict=True) into a tidy DataFrame.
    Keeps class rows first, then averages and accuracy at the bottom.
    """
    df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "label"})
    # Move accuracy row to the bottom (it has only 'precision' filled as accuracy in sklearn)
    # Ensure consistent column order even when some metrics are missing
    want_cols = ["label", "precision", "recall", "f1-score", "support"]
    for c in want_cols:
        if c not in df.columns:
            df[c] = np.nan
    # Make label order: class rows (those which are numeric or not containing 'avg'/'accuracy'),
    # then macro avg, weighted avg, micro avg (if present), then accuracy
    def _is_avg(x): return any(k in x for k in ["avg", "accuracy"])
    class_rows = df[~df["label"].apply(_is_avg)]
    other_rows = df[df["label"].apply(_is_avg)]

    # Custom order: micro avg, macro avg, weighted avg, accuracy
    order_map = {"micro avg": 0, "macro avg": 1, "weighted avg": 2, "accuracy": 3}
    other_rows["order"] = other_rows["label"].map(order_map).fillna(99)
    other_rows = other_rows.sort_values("order").drop(columns=["order"])

    tidy = pd.concat([class_rows, other_rows], axis=0, ignore_index=True)
    # Nice numeric formatting
    for c in ["precision", "recall", "f1-score"]:
        tidy[c] = pd.to_numeric(tidy[c], errors="coerce")
    tidy["support"] = pd.to_numeric(tidy["support"], errors="coerce").astype("Int64")
    return tidy[want_cols]

def _download_button_df(df: pd.DataFrame, label: str, filename: str, help_text: str = ""):
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        help=help_text,
        use_container_width=True
    )

# ---------- Load data & create samples ----------
X, y = load_dataset()
ensure_dirs()
p_feat, p_feat_target = make_test_csv(pd.DataFrame(X), y, n_rows=50)

# ---------- Config controls (MAIN area; no sidebar) ----------
st.subheader("Configuration")
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    seed = st.number_input("Random seed", value=42, step=1)
with c2:
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
with c3:
    use_mnb = st.checkbox("Use MultinomialNB (for count features)", value=False)
with c4:
    best_metric = st.selectbox(
        "Pick best model by metric",
        ["AUC", "F1", "Accuracy", "Precision", "Recall", "MCC"],
        index=0
    )

# Quick downloads in main
with st.expander("⬇️ Quick downloads"):
    b1 = _read_bytes(p_feat)
    b2 = _read_bytes(p_feat_target)
    cols_dl = st.columns(2)
    with cols_dl[0]:
        if b1:
            st.download_button(
                "Download test_data.csv (features only)",
                data=b1, file_name="test_data.csv", mime="text/csv",
                use_container_width=True
            )
    with cols_dl[1]:
        if b2:
            st.download_button(
                "Download test_data_with_target.csv",
                data=b2, file_name="test_data_with_target.csv", mime="text/csv",
                use_container_width=True
            )

# ---------- Dataset preview ----------
st.subheader("Dataset snapshot")
st.dataframe(pd.DataFrame(X).head(), use_container_width=True)
st.caption(f"Instances: {X.shape[0]} • Features: {X.shape[1]}")

# ---------- File uploader ----------
uploaded = st.file_uploader("Upload test CSV", type=["csv"])
st.markdown(
    "- ✅ CSV **with** a **target** column → full evaluation (metrics)\n"
    "- 🔎 CSV **without** **target** → predictions only + **predictions.csv** download"
)

# Read uploaded file ONCE and reuse
uploaded_df = None
has_target = False
if uploaded is not None:
    # Guard: truly empty file
    if getattr(uploaded, "size", None) == 0:
        st.error("The uploaded file is empty. Please select a non-empty CSV.")
        st.stop()
    try:
        uploaded.seek(0)  # always rewind before first read
        uploaded_df = pd.read_csv(uploaded)
    except pd.errors.EmptyDataError:
        st.error("No columns or data found in the uploaded file. Please upload a valid CSV.")
        st.stop()
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        st.stop()
    has_target = "target" in uploaded_df.columns

# ---------- Train & evaluate on default holdout ----------
cols, results, fitted, (X_test, y_test) = run_experiment(
    test_size=float(test_size), random_state=int(seed), mnb=bool(use_mnb)
)

# ---------- If uploaded CSV present: align columns and, if available, use target ----------
if uploaded_df is not None:
    # Align by feature names; fill missing with 0
    X_aligned = uploaded_df.reindex(columns=cols, fill_value=0)
    st.success(f"Custom CSV loaded. Columns aligned to model features ({len(cols)}).")
    with st.expander("Preview: uploaded (aligned) features"):
        st.dataframe(X_aligned.head(), use_container_width=True)
    if has_target:
        # Use user-supplied labels for evaluation
        y_uploaded = uploaded_df["target"]
        X_test = X_aligned
        y_test = y_uploaded

# ---------- Evaluate all models on current test set (holdout or uploaded) ----------
from collections import OrderedDict
eval_results = OrderedDict()
for name, mdl in fitted.items():
    res = evaluate_model(mdl, X_test, y_test)
    eval_results[name] = res

# ---------- TABS ----------
tab_overview, tab_compare, tab_detail, tab_preds = st.tabs(
    ["Overview", "Model Comparison", "Detailed View", "Predictions"]
)

with tab_overview:
    # Best model & selector
    best_name = pick_best_model(eval_results, metric=best_metric)
    st.success(f"🏅 Best model by **{best_metric}**: **{best_name}**")
    # Small KPI strip for the best model
    res_best = eval_results[best_name]
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Accuracy", f"{res_best.metrics.get('Accuracy', np.nan):.4f}")
    k2.metric("AUC", f"{res_best.metrics.get('AUC', np.nan):.4f}")
    k3.metric("Precision", f"{res_best.metrics.get('Precision', np.nan):.4f}")
    k4.metric("Recall", f"{res_best.metrics.get('Recall', np.nan):.4f}")
    k5.metric("F1", f"{res_best.metrics.get('F1', np.nan):.4f}")
    k6.metric("MCC", f"{res_best.metrics.get('MCC', np.nan):.4f}")

with tab_compare:
    st.subheader("Comparison Table – Metrics")
    rows = []
    for name, res in eval_results.items():
        row = {"ML Model Name": name}
        row.update(res.metrics)
        rows.append(row)
    df_metrics = pd.DataFrame(rows)
    df_metrics = df_metrics.reindex(
        columns=["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    )
    st.dataframe(df_metrics, use_container_width=True)
    _download_button_df(df_metrics, "Download metrics.csv", "metrics.csv", "All models' metrics")

with tab_detail:
    st.subheader("Detailed View")
    choice = st.selectbox(
        "Inspect model",
        list(eval_results.keys()),
        index=list(eval_results.keys()).index(pick_best_model(eval_results, metric=best_metric))
    )
    res = eval_results[choice]

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots()
        sns.heatmap(res.confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

    with col2:
        st.markdown("**Classification Report**")
        # Build a neat table from sklearn's text/structure if available
        report_df = None
        if hasattr(res, "report_dict") and isinstance(res.report_dict, dict):
            report_df = _format_classification_report(res.report_dict)
        else:
            # Fallback: parse from text if only string provided
            try:
                from sklearn.metrics import classification_report
                # Recompute to ensure we have output_dict
                # (We don't know the model's classes ordering from res; recompute safe)
                y_pred_tmp = res.y_pred if hasattr(res, "y_pred") else None
                # If res doesn't store y_pred, recompute from fitted model:
                # Find selected model and recompute:
                mdl = fitted[choice]
                y_pred_tmp = mdl.predict(X_test)
                rep_dict = classification_report(y_test, y_pred_tmp, output_dict=True, zero_division=0)
                report_df = _format_classification_report(rep_dict)
            except Exception:
                report_df = pd.DataFrame({"message": ["Could not build report table"]})

        st.dataframe(
            report_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}"}),
            use_container_width=True
        )
        _download_button_df(report_df, "Download classification_report.csv", "classification_report.csv")

with st.expander("💾 Model files & downloads"):
    # Save & download models
    ensure_dirs()
    best_name_for_file = pick_best_model(eval_results, metric=best_metric)
    best_model_path = save_model_pkl(fitted[best_name_for_file], out_path="model/saved/best_model.pkl")
    best_bytes = _read_bytes(best_model_path)
    cdl1, cdl2 = st.columns(2)
    with cdl1:
        if best_bytes:
            st.download_button(
                "Download best_model.pkl",
                data=best_bytes,
                file_name="best_model.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
    with cdl2:
        if st.button("Save selected model as .pkl"):
            # Use current selected choice if user visited Detailed View; otherwise default best
            choice_for_save = 'choice' in locals() and choice or best_name_for_file
            safe = "".join(c for c in choice_for_save if c.isalnum() or c in ("_", "-")).strip("_-")
            sel_path = save_model_pkl(fitted[choice_for_save], out_path=f"model/saved/{safe}.pkl")
            st.success(f"Saved: {sel_path}")

    # Expose selected model download if present
    choice_for_download = 'choice' in locals() and choice or best_name_for_file
    sel_safe = "".join(c for c in choice_for_download if c.isalnum() or c in ("_", "-")).strip("_-")
    sel_bytes = _read_bytes(f"model/saved/{sel_safe}.pkl")
    if sel_bytes:
        st.download_button(
            "Download selected model .pkl",
            data=sel_bytes,
            file_name=f"{choice_for_download.replace(' ', '_')}.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )

with tab_preds:
    # Predictions if uploaded CSV has NO target
    if uploaded_df is not None and not has_target:
        st.subheader("Predictions (best model) for uploaded CSV (no target provided)")
        mdl = fitted[pick_best_model(eval_results, metric=best_metric)]
        # Reuse the already-read & aligned frame
        X_aligned = uploaded_df.reindex(columns=cols, fill_value=0)
        y_pred = mdl.predict(X_aligned)
        preds_df = uploaded_df.copy()
        preds_df["prediction"] = y_pred
        if hasattr(mdl, "predict_proba"):
            try:
                proba = mdl.predict_proba(X_aligned)
                preds_df["prob_1"] = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else np.nan
            except Exception:
                pass
        st.dataframe(preds_df.head(), use_container_width=True)
        st.download_button(
            "Download predictions.csv",
            data=preds_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Upload a CSV **without** a `target` column to get prediction outputs here.")
