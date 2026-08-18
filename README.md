# Machine Learning Assignment 2 — WDBC Classification

## a. Problem Statement
Build and compare five classification models for predicting whether a breast tumor is benign or malignant using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Evaluate each model using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC), and demonstrate the models through an interactive Streamlit application.

## b. Dataset Description
Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), from the University of Wisconsin.

- Instances: 569
- Input features: 30 real-valued features
- Target: diagnosis
  - 0 = Benign
  - 1 = Malignant
- Missing values: none

## c. Github Repository Link
**GitHub Repository:** `https://github.com/vikramdurgude/ML_Assignment_2.git`

## d. Models Used
The five required models are:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (kNN)
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest | 0.9737 | 0.9944 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |


### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong baseline with high accuracy and AUC, giving balanced precision and recall. |
| Decision Tree | Good performance, but lower than the other stronger models on this test split. |
| kNN | Strong performance after standardization, with good accuracy and balanced classification results. |
| Naive Bayes | High AUC and perfect precision on this split, although recall is lower than the top models. |
| Random Forest (Ensemble) | Best overall performance on this test split, achieving the highest accuracy, F1 and MCC. |

### Overall Winner
**Random Forest**

On the fixed 80:20 stratified test split (`random_state=42`), Random Forest achieved the best overall accuracy.

## Streamlit Application Features
1. CSV test-data upload.
2. Model-selection dropdown.
3. Accuracy, AUC, Precision, Recall, F1 and MCC display.
4. Confusion matrix.
5. Classification report.
6. Prediction table and downloadable predictions CSV.

## Streamlit App Link
**Live Streamlit App:** `https://mlassignment2-xgq9ohsmwsrpe5bn7ly8fy.streamlit.app/`


## Reproducibility
- Train/test split: 80/20
- Stratified split
- Random state: 42
- Logistic Regression and kNN use StandardScaler.
- Random Forest uses 300 trees.
- kNN uses 7 neighbors.
- Decision Tree uses maximum depth 5.
