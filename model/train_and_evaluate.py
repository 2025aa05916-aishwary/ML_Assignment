# model/train_and_evaluate.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    confusion: np.ndarray
    report: str


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


def get_models(mnb: bool = False) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes": MultinomialNB() if mnb else GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=42
        ),
    }

    if _HAS_XGB:
        models["XGBoost (Ensemble)"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False
        )
    else:
        models["Gradient Boosting (Fallback to XGB)"] = GradientBoostingClassifier(
            random_state=42
        )

    return models


def _auc_for(model, X_test, y_test) -> float:
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return float(roc_auc_score(y_test, proba[:, 1]))
            else:
                return float(roc_auc_score(y_test, proba, multi_class="ovr"))
        if hasattr(model, "decision_function"):
            df = model.decision_function(X_test)
            if df.ndim == 1:
                df_min, df_max = float(df.min()), float(df.max())
                scr = (df - df_min) / (df_max - df_min + 1e-12)
                return float(roc_auc_score(y_test, scr))
            else:
                return float(roc_auc_score(y_test, df, multi_class="ovr"))
    except Exception:
        pass
    return float("nan")


def evaluate_model(model, X_test, y_test) -> EvaluationResult:
    y_pred = model.predict(X_test)
    avg = "binary" if len(np.unique(y_test)) == 2 else "weighted"
    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
    }
    metrics["AUC"] = _auc_for(model, X_test, y_test)
    conf = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return EvaluationResult(metrics=metrics, confusion=conf, report=report)


def ensure_dirs():
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("model/saved").mkdir(parents=True, exist_ok=True)


def make_test_csv(X: pd.DataFrame, y: pd.Series, n_rows: int = 50):
    ensure_dirs()
    n = min(n_rows, len(X))
    idx = np.random.RandomState(42).choice(len(X), size=n, replace=False)
    Xs = X.iloc[idx].copy()
    ys = y.iloc[idx].copy()

    p1 = "data/test_data.csv"
    p2 = "data/test_data_with_target.csv"
    Xs.to_csv(p1, index=False)
    pd.concat([Xs, ys.rename("target")], axis=1).to_csv(p2, index=False)
    return p1, p2


def save_model_pkl(model, out_path: str = "model/saved/best_model.pkl") -> str:
    ensure_dirs()
    joblib.dump(model, out_path)
    return out_path


def pick_best_model(results: Dict[str, EvaluationResult], metric: str = "AUC") -> str:
    metric = metric if metric in {"AUC", "Accuracy", "Precision", "Recall", "F1", "MCC"} else "AUC"
    series = pd.Series({k: v.metrics.get(metric, np.nan) for k, v in results.items()})
    if series.isna().all() and metric == "AUC":
        series = pd.Series({k: v.metrics.get("F1", np.nan) for k, v in results.items()})
    return series.sort_values(ascending=False).index[0]


def run_experiment(test_size: float = 0.2,
                   random_state: int = 42,
                   mnb: bool = False):
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    models = get_models(mnb=mnb)
    results: Dict[str, EvaluationResult] = {}
    fitted: Dict[str, Any] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        res = evaluate_model(model, X_test, y_test)
        results[name] = res
        fitted[name] = model

    make_test_csv(X_test, y_test, n_rows=50)
    return X.columns.tolist(), results, fitted, (X_test, y_test)


if __name__ == "__main__":
    cols, results, fitted, holdout = run_experiment()
    print("Columns:", cols)
    for k, v in results.items():
        print("\n==", k)
        print(v.metrics)
        print("Confusion:\n", v.confusion)
        print("Report:\n", v.report)
