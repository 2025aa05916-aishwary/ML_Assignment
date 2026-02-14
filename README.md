# End‑to‑End Classification – ML Assignment 2

## Problem Statement
This project addresses the problem of classifying breast cancer as either benign or malignant based on various cell nucleus characteristics. Early and accurate diagnosis is crucial for effective treatment, making this a critical application of machine learning.

## Dataset Description
**Dataset**: The dataset used for this classification task is the Breast Cancer Wisconsin (Diagnostic) dataset, which is publicly available through scikit-learn. It contains features computed from digitized images of fine needle aspirates (FNAs) of breast masses. Each instance represents characteristics of a cell nucleus, and the goal is to predict whether the mass is benign (0) or malignant (1).

## Models Used
1. Logistic Regression  
2. Decision Tree Classifier  
3. K‑Nearest Neighbors (kNN)  
4. Naive Bayes (GaussianNB by default; MultinomialNB optional)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Comparison Table (auto‑generated in app)

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.982456	0.99537	0.986111	0.986111	0.986111	0.962302
Decision Tree	0.912281	0.915675	0.955882	0.902778	0.928571	0.817412
kNN	0.973684	0.988426	0.96	1	0.979592	0.944155
Naive Bayes	0.938596	0.987765	0.945205	0.958333	0.951724	0.867553
Random Forest (Ensemble)	0.947368	0.993717	0.958333	0.958333	0.958333	0.886905
XGBoost (Ensemble)	0.95614	0.99504	0.946667	0.986111	0.965986	0.905824
<img width="790" height="204" alt="image" src="https://github.com/user-attachments/assets/75c15bd7-9678-4b9b-bcd2-12e24256c717" />


## Observations (fill after running)

ML Model Name	Observation about model performance
Logistic Regression	Achieved very high accuracy and AUC, indicating excellent overall performance and discrimination ability.
Decision Tree	Exhibited decent performance but lower than other models, potentially prone to overfitting if not properly pruned or tuned.
kNN	Demonstrated perfect recall, meaning it correctly identified all positive cases, with strong overall accuracy. However, precision is slightly lower.
Naive Bayes	Showed good performance across metrics, particularly a high AUC, suggesting good separability between classes.
Random Forest (Ensemble)	Provided strong and consistent performance across all metrics, benefiting from the ensemble approach to reduce variance.
XGBoost (Ensemble)	Achieved excellent and balanced performance with high AUC and F1 score, making it one of the top-performing models in this comparison.




