import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model.train_and_evaluate import run_experiment, load_dataset, get_models, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="ML Assignment 2 – Classification Bench", layout='wide')
st.title("ML Assignment 2 – End‑to‑End Classification App")
st.markdown("This app demonstrates six classifiers, evaluation metrics, and visualization for the Breast Cancer Wisconsin dataset. Upload **test CSV** optionally to evaluate on your own split as per assignment.")

# Sidebar controls
st.sidebar.header("Configuration")
seed = st.sidebar.number_input("Random seed", value=42, step=1)
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
use_mnb = st.sidebar.checkbox("Use MultinomialNB instead of GaussianNB (for count features)", value=False)

# Data loading
X, y = load_dataset()
st.subheader("Dataset snapshot")
st.write(pd.DataFrame(X).head())
st.write(f"Instances: {X.shape[0]} | Features: {X.shape[1]}")

# Optional test CSV upload (test only)
upload = st.file_uploader("Upload **test** CSV (columns must match scikit-learn Breast Cancer feature names)", type=['csv'])

# Prepare split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X), y, test_size=test_size, random_state=seed, stratify=y)

if upload is not None:
    try:
        test_df = pd.read_csv(upload)
        # Align columns
        test_df = test_df.reindex(columns=X_train.columns, fill_value=0)
        X_test = test_df
        st.success("Custom test CSV loaded and aligned with feature columns.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# Models
models = get_models(mnb=use_mnb)
model_names = list(models.keys())
choice = st.sidebar.selectbox("Select model to inspect", model_names)

# Train all, evaluate all
from collections import OrderedDict
results = OrderedDict()
for name, m in models.items():
    m.fit(X_train, y_train)
    res = evaluate_model(m, X_test, y_test if isinstance(y_test, np.ndarray) else y_test)
    results[name] = res

# Summary table
st.subheader("Comparison Table – Metrics")
rows = []
for name, res in results.items():
    r = {'ML Model Name': name}
    r.update(res.metrics)
    rows.append(r)
df_metrics = pd.DataFrame(rows)
df_metrics = df_metrics[['ML Model Name','Accuracy','AUC','Precision','Recall','F1','MCC']]
st.dataframe(df_metrics, use_container_width=True)

# Selected model details
st.subheader(f"Details – {choice}")
res = results[choice]
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Confusion Matrix**")
    fig, ax = plt.subplots()
    sns.heatmap(res.confusion, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)
with col2:
    st.markdown("**Classification Report**")
    st.text(res.report)

st.caption("Built for BITS ML Assignment‑2: includes dataset upload, model selection, metrics, and confusion matrix/classification report.")
