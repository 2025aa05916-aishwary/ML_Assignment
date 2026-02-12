# model/train_and_evaluate.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple
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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    confusion: np.ndarray
    report: str


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y


def get_models(mnb: bool=False) -> Dict[str, Any]:
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, n_jobs=None))
        ]),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'kNN': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=7))
        ]),
        'Naive Bayes': GaussianNB() if not mnb else MultinomialNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=300, random_state=42),
        'XGBoost (Ensemble)': XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )
    }
    return models


def evaluate_model(model, X_test, y_test, y_prob=None) -> EvaluationResult:
    y_pred = model.predict(X_test)
    # compute probabilities if available
    if y_prob is None:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:,1]
        elif hasattr(model, 'decision_function'):
            # scale decision function to 0-1 via min-max
            df = model.decision_function(X_test)
            df_min, df_max = df.min(), df.max()
            y_prob = (df - df_min) / (df_max - df_min + 1e-12)
        else:
            y_prob = None
    average = 'binary' if len(np.unique(y_test))==2 else 'weighted'
    metrics = {
        'Accuracy': float(accuracy_score(y_test, y_pred)),
        'Precision': float(precision_score(y_test, y_pred, average=average, zero_division=0)),
        'Recall': float(recall_score(y_test, y_pred, average=average, zero_division=0)),
        'F1': float(f1_score(y_test, y_pred, average=average, zero_division=0)),
        'MCC': float(matthews_corrcoef(y_test, y_pred))
    }
    # AUC only when probabilities available
    if y_prob is not None:
        try:
            if len(np.unique(y_test))==2:
                metrics['AUC'] = float(roc_auc_score(y_test, y_prob))
            else:
                metrics['AUC'] = float(roc_auc_score(y_test, y_prob, multi_class='ovr'))
        except Exception:
            metrics['AUC'] = float('nan')
    else:
        metrics['AUC'] = float('nan')
    conf = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return EvaluationResult(metrics=metrics, confusion=conf, report=report)


def run_experiment(test_size: float=0.2, random_state: int=42, mnb: bool=False):
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    models = get_models(mnb)
    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        res = evaluate_model(model, X_test, y_test)
        results[name] = res
        fitted[name] = model
    return X.columns.tolist(), results, fitted, (X_test, y_test)

if __name__ == '__main__':
    cols, results, fitted, holdout = run_experiment()
    print('Columns:', cols)
    for k,v in results.items():
        print('\n==', k)
        print(v.metrics)
        print('Confusion:\n', v.confusion)
        print('Report:\n', v.report)
