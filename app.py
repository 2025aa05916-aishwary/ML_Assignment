# app.py
import os
from io import BytesIO

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from model.train_and_evaluate import (
    run_experiment, load_dataset, get_models, evaluate_model,
    make_test_csv, save_model_pkl, pick_best_model, ensure_dirs
)

st.set_page_config(page_title="ML Assignment 2 – Classification Bench", layout="wide")
st.title("ML Assignment 2 – End‑to‑End Classification App")
st.markdown(
    "This app trains **six classifiers** on the Breast Cancer Wisconsin dataset, "
    "compares metrics, and lets you **upload a test CSV** and **download** a sample test CSV "
    "and the **best model (.pkl)**."
)

# Sidebar controls
st.sidebar.header("Configuration")
seed = st.sidebar.number_input("Random seed", value=42, step=1)
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
use_mnb = st.sidebar.checkbox("Use MultinomialNB (for count features)", value=False)
best_metric = st.sidebar.selectbox(
    "Pick best model by metric", ["AUC", "F1", "Accuracy", "Precision", "Recall", "MCC"], index=0
)

# Data preview
X, y = load_dataset()
st.subheader("Dataset snapshot")
st.dataframe(pd.DataFrame(X).head(), use_container_width=True)
st.caption(f"Instances: {X.shape[0]} • Features: {X.shape[1]}")

# Create sample CSVs
ensure_dirs()
p_feat, p_feat_target = make_test_csv(pd.DataFrame(X), y, n_rows=50)

def _read_bytes(path: str) -> bytes | None:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

# Sidebar downloads
st.sidebar.subheader("⬇️ Quick downloads")
b1 = _read_bytes(p_feat)
b2 = _read_bytes(p_feat_target)
if b1:
    st.sidebar.download_button(
        "Download test_data.csv (features only)",
        data=b1, file_name="test_data.csv", mime="text/csv", use_container_width=True
    )
if b2:
    st.sidebar.download_button(
        "Download test_data_with_target.csv",
        data=b2, file_name="test_data_with_target.csv", mime="text/csv", use_container_width=True
    )

# Optional user CSV
uploaded = st.file_uploader(
    "Upload CSV to use as **test** data.\n"
    "- If you include a `target` column (0/1), we will evaluate metrics on it.\n"
    "- If no `target` column, we will only show predictions and scores where possible.",
    type=["csv"]
)

# Train & evaluate on holdout
cols, results, fitted, (X_test, y_test) = run_experiment(
    test_size=float(test_size), random_state=int(seed), mnb=bool(use_mnb)
)

# If user uploaded a CSV, align & optionally evaluate
if uploaded is not None:
    try:
        test_df = pd.read_csv(uploaded)
        has_target = "target" in test_df.columns
        X_aligned = test_df.reindex(columns=cols, fill_value=0)
        st.success(f"Custom CSV loaded. Columns aligned to model features ({len(cols)}).")

        if has_target:
            y_uploaded = test_df["target"]
            X_test = X_aligned
            y_test = y_uploaded

        with st.expander("Preview: uploaded (aligned) features"):
            st.dataframe(X_aligned.head(), use_container_width=True)

    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")

# Evaluate all models on current test set
from collections import OrderedDict
eval_results = OrderedDict()
for name, mdl in fitted.items():
    res = evaluate_model(mdl, X_test, y_test)
    eval_results[name] = res

# Comparison table
st.subheader("Comparison Table – Metrics")
rows = []
for name, res in eval_results.items():
    row = {"ML Model Name": name}
    row.update(res.metrics)
    rows.append(row)
df_metrics = pd.DataFrame(rows)
df_metrics = df_metrics.reindex(columns=["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])
st.dataframe(df_metrics, use_container_width=True)

# Best model
best_name = pick_best_model(eval_results, metric=best_metric)
st.success(f"🏅 Best model by **{best_metric}**: **{best_name}**")

# Model details
choice = st.selectbox("Inspect model", list(eval_results.keys()),
                      index=list(eval_results.keys()).index(best_name))
res = eval_results[choice]

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Confusion Matrix**")
    fig, ax = plt.subplots()
    sns.heatmap(res.confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)
with col2:
    st.markdown("**Classification Report**")
    st.text(res.report)

# Save & download models
ensure_dirs()
best_model_path = save_model_pkl(fitted[best_name], out_path="model/saved/best_model.pkl")
best_bytes = _read_bytes(best_model_path)
if best_bytes:
    st.sidebar.download_button(
        "Download best_model.pkl",
        data=best_bytes,
        file_name="best_model.pkl",
        mime="application/octet-stream",
        use_container_width=True
    )

if st.sidebar.button("💾 Save selected model as .pkl"):
    safe = "".join(c for c in choice if c.isalnum() or c in ("_", "-")).strip("_-")
    sel_path = save_model_pkl(fitted[choice], out_path=f"model/saved/{safe}.pkl")
    st.sidebar.success(f"Saved: {sel_path}")

sel_bytes = _read_bytes("model/saved/{}.pkl".format("".join(c for c in choice if c.isalnum() or c in ("_", "-")).strip("_-")))
if sel_bytes:
    st.sidebar.download_button(
        "Download selected model .pkl",
        data=sel_bytes,
        file_name=f"{choice.replace(' ', '_')}.pkl",
        mime="application/octet-stream",
        use_container_width=True
    )

# If uploaded had no target → predictions only
if uploaded is not None and "target" not in pd.read_csv(uploaded, nrows=1).columns:
    st.subheader("Predictions (best model) for uploaded CSV (no target provided)")
    mdl = fitted[best_name]
    uploaded_full = pd.read_csv(uploaded)
    X_aligned = uploaded_full.reindex(columns=cols, fill_value=0)
    y_pred = mdl.predict(X_aligned)
    preds_df = uploaded_full.copy()
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
    )
