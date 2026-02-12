# End‑to‑End Classification – ML Assignment 2

## Problem Statement
Build six classification models on a public dataset, report comprehensive metrics, and deploy an interactive Streamlit app.

## Dataset Description
**Dataset**: Breast Cancer Wisconsin (Diagnostic) – from `sklearn.datasets.load_breast_cancer()`.
- Instances: 569
- Features: 30 real‑valued features derived from digitized images of fine needle aspirates of breast masses.
- Target: Binary (0 = malignant, 1 = benign)

## Models Used
1. Logistic Regression  
2. Decision Tree Classifier  
3. K‑Nearest Neighbors (kNN)  
4. Naive Bayes (GaussianNB by default; MultinomialNB optional)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Comparison Table (auto‑generated in app)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression |  |  |  |  |  |  |
| Decision Tree |  |  |  |  |  |  |
| kNN |  |  |  |  |  |  |
| Naive Bayes |  |  |  |  |  |  |
| Random Forest (Ensemble) |  |  |  |  |  |  |
| XGBoost (Ensemble) |  |  |  |  |  |  |

## Observations (fill after running)
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| kNN | |
| Naive Bayes | |
| Random Forest (Ensemble) | |
| XGBoost (Ensemble) | |

## Repository Structure
```
project/
├── app.py
├── requirements.txt
├── README.md
└── model/
    └── train_and_evaluate.py
```

## How to Run Locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Deployment (Community Cloud)
1. Push this repo to GitHub.
2. Go to https://streamlit.io/cloud → **New app**.
3. Select the repo, branch (main), and `app.py`. Click **Deploy**.

## Notes
- The app allows uploading **test CSV** (only) due to free‑tier resource limits.
- AUC may show `NaN` when a model cannot compute probabilities.
- MCC is computed for all models.
