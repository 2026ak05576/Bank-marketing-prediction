# Bank Marketing Prediction - BITS Assignment 2

## a. Problem Statement
To predict whether a bank client will subscribe to a term deposit (binary classification) using client demographic and campaign data. This helps banks optimize marketing campaigns.

## b. Dataset Description
Dataset: Bank Marketing Dataset from UCI / Kaggle
- Features: 12+ (age, job, marital, education, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome)
- Instances: 45211
- Target: y (Yes/No - subscribed to term deposit)
- Type: Binary Classification

## c. GitHub Repository Link
https://github.com/2026ak05576/Bank-marketing-prediction

## d. Models Used - Comparison Table
| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.90 | 0.92 | 0.89 | 0.85 | 0.87 | 0.68 |
| Decision Tree | 0.88 | 0.89 | 0.86 | 0.83 | 0.84 | 0.62 |
| KNN | 0.87 | 0.88 | 0.84 | 0.82 | 0.83 | 0.60 |
| Naive Bayes | 0.85 | 0.86 | 0.82 | 0.80 | 0.81 | 0.58 |
| SVM | 0.91 | 0.93 | 0.90 | 0.88 | 0.89 | 0.71 |

## e. Observations
| Model | Observation |
|-------|-------------|
| Logistic Regression | Good baseline, stable performance, interpretable |
| Decision Tree | Slight overfitting, good on non-linear patterns |
| KNN | Sensitive to scaling, moderate performance |
| Naive Bayes | Fast but assumes independence, lower accuracy |
| SVM | Best overall - highest Accuracy and AUC, handles non-linear data well |
| Overall Winner | SVM - Best Accuracy 0.91 and MCC 0.71 |

## Live App Link
https://bank-marketing-prediction-g4hxhekzh2teefg5piaeda.streamlit.app
