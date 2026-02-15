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
    save_model_pkl, pick_best_model, ensure_dirs
)

# ---------- Page config ----------
st.set_page_config(page_title="ML Assignment 2 – Classification Bench", layout="wide")
st.title("ML Assignment 2 – End‑to‑End Classification App")
st.markdown(
    "This app trains **six classifiers** on the Breast Cancer Wisconsin dataset, "
    "compares metrics, lets you **upload a test CSV**, and **download** a single full "
    "test CSV (features only, 569 rows) and the **best model (.pkl)**."
)

# ---------- Helpers ----------
def _read_bytes(path: str) -> Optional[bytes]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

def _format_classification_report(report_dict: Dict[str, Any], n_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Turn sklearn classification_report(output_dict=True) into a tidy DataFrame.
    Keeps class rows first, then micro/macro/weighted averages, then accuracy.
    Robust handling of 'support' so we never fail casting.
    """
    df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "label"})
    # Ensure consistent columns exist
    desired_cols = ["label", "precision", "recall", "f1-score", "support"]
    for c in desired_cols:
        if c not in df.columns:
            df[c] = np.nan

    # If 'accuracy' row exists, set its support to total samples (or NA if unknown)
    if "accuracy" in df["label"].values:
        idx = df.index[df["label"] == "accuracy"]
        if len(idx) > 0:
            if n_samples is not None:
                df.loc[idx, "support"] = n_samples
            else:
                df.loc[idx, "support"] = np.nan

    # Split class rows vs aggregation rows
    agg_labels = ("accuracy", "macro avg", "micro avg", "weighted avg")
    class_rows = df[~df["label"].isin(agg_labels)]
    agg_rows = df[df["label"].isin(agg_labels)].copy()

    # Order aggregation rows
    order_map = {"micro avg": 0, "macro avg": 1, "weighted avg": 2, "accuracy": 3}
    agg_rows["order"] = agg_rows["label"].map(order_map).fillna(99)
    agg_rows = agg_rows.sort_values("order").drop(columns=["order"])

    out = pd.concat([class_rows, agg_rows], ignore_index=True)

    # Numeric conversions
    for c in ["precision", "recall", "f1-score"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Handle 'support' safely:
    support_num = pd.to_numeric(out["support"], errors="coerce")
    is_integer_like = np.isfinite(support_num) & (np.floor(support_num) == support_num)
    support_int = pd.Series(pd.NA, index=out.index, dtype="Int64")
    support_int[is_integer_like] = support_num[is_integer_like].astype("Int64")
    out["support"] = support_int

    return out[desired_cols]

def _download_button_df(df: pd.DataFrame, label: str, filename: str, help_text: str = ""):
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        help=help_text,
        use_container_width=True
    )

# ---------- Load data ----------
X, y = load_dataset()

# ---------- Configuration (MAIN area; no sidebar) ----------
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

# ---------- Dataset preview ----------
st.subheader("Dataset snapshot")
st.dataframe(pd.DataFrame(X).head(), use_container_width=True)
st.caption(f"Instances: {pd.DataFrame(X).shape[0]} • Features: {pd.DataFrame(X).shape[1]}")

# ---------- Sidebar: single full features-only CSV download ----------
full_features_csv_bytes = pd.DataFrame(X).to_csv(index=False).encode("utf-8")
st.sidebar.subheader("⬇️ Download Test CSV")
st.sidebar.download_button(
    "Download FULL test_data.csv (569 rows)",
    data=full_features_csv_bytes,
    file_name="test_data_full.csv",
    mime="text/csv",
    use_container_width=True
)

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
    if getattr(uploaded, "size", None) == 0:
        st.error("The uploaded file is empty. Please select a non-empty CSV.")
        st.stop()
    try:
        uploaded.seek(0)
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
    X_aligned = uploaded_df.reindex(columns=cols, fill_value=0)
    st.success(f"Custom CSV loaded. Columns aligned to model features ({len(cols)}).")
    with st.expander("Preview: uploaded (aligned) features"):
        st.dataframe(X_aligned.head(), use_container_width=True)
    if has_target:
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
    ["Overview", "Evaluation Metrics", "Detailed View", "Predictions"]
)

with tab_overview:
    # Best model & selector
    best_name = pick_best_model(eval_results, metric=best_metric)
    st.success(f"🏅 Best model by **{best_metric}**: **{best_name}**")
    res_best = eval_results[best_name]

    # KPI strip titled Evaluation Metrics
    st.markdown("### Evaluation Metrics")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Accuracy", f"{res_best.metrics.get('Accuracy', np.nan):.4f}")
    k2.metric("AUC", f"{res_best.metrics.get('AUC', np.nan):.4f}")
    k3.metric("Precision", f"{res_best.metrics.get('Precision', np.nan):.4f}")
    k4.metric("Recall", f"{res_best.metrics.get('Recall', np.nan):.4f}")
    k5.metric("F1", f"{res_best.metrics.get('F1', np.nan):.4f}")
    k6.metric("MCC", f"{res_best.metrics.get('MCC', np.nan):.4f}")

with tab_compare:
    st.subheader("Evaluation Metrics")
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
    default_idx = list(eval_results.keys()).index(pick_best_model(eval_results, metric=best_metric))
    choice = st.selectbox("Inspect model", list(eval_results.keys()), index=default_idx)
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
        # Always recompute safely to guarantee a table
        from sklearn.metrics import classification_report
        mdl = fitted[choice]
        try:
            y_pred = mdl.predict(X_test)
            rep_dict = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )
            n_samples = len(y_test) if y_test is not None else None
            report_df = _format_classification_report(rep_dict, n_samples=n_samples)
        except Exception as e:
            report_df = pd.DataFrame(
                {"label": ["error"], "precision": [np.nan], "recall": [np.nan],
                 "f1-score": [np.nan], "support": [pd.NA]}
            )
            st.warning(f"Could not build report table: {e}")

        st.dataframe(
            report_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}"}),
            use_container_width=True
        )
        _download_button_df(report_df, "Download classification_report.csv", "classification_report.csv")

with st.expander("💾 Model files & downloads"):
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
            choice_for_save = 'choice' in locals() and choice or best_name_for_file
            safe = "".join(c for c in choice_for_save if c.isalnum() or c in ("_", "-")).strip("_-")
            sel_path = save_model_pkl(fitted[choice_for_save], out_path=f"model/saved/{safe}.pkl")
            st.success(f"Saved: {sel_path}")

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
    if uploaded_df is not None and not has_target:
        st.subheader("Predictions (best model) for uploaded CSV (no target provided)")
        mdl = fitted[pick_best_model(eval_results, metric=best_metric)]
        X_aligned = uploaded_df.reindex(columns=cols, fill_value=0)
        y_pred = mdl.predict(X_aligned)

        preds_df = uploaded_df.copy()
        preds_df["prediction"] = y_pred

        if hasattr(mdl, "predict_proba"):
            try:
                proba = mdl.predict_proba(X_aligned)
                preds_df["prob_1"] = (
                    proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else np.nan
                )
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
