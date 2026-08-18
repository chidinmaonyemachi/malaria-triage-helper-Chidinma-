# 🦟 Malaria Symptom Triage Helper


## Live Application

 **Streamlit App:** https://tsfojyhrk6dq3dmk6tpedb.streamlit.app

## Project Overview

The Malaria Symptom Triage Helper is an educational AI/ML project that uses machine learning to analyse basic patient information and self-reported symptoms associated with malaria.

The application provides a malaria-like symptom pattern assessment and triage guidance.

This project is intended for educational demonstration only and is not a medical diagnostic system.

---

## Problem Statement

Malaria symptoms such as fever, chills, headache, vomiting, weakness, and diarrhea can overlap with symptoms of other illnesses.

This project explores how machine learning can be used to identify patterns in symptom data and provide an educational triage recommendation.

The system does not diagnose malaria. Users with symptoms are encouraged to seek appropriate medical assessment and malaria testing.

---

## Project Objectives

The objectives of this project are to:

- Analyse a malaria-related symptom dataset.
- Perform data cleaning and exploratory data analysis.
- Select relevant features for machine-learning prediction.
- Train and compare classification algorithms.
- Evaluate model performance using multiple metrics.
- Build an interactive Streamlit application.
- Include a safety-based triage layer for serious warning signs.
- Demonstrate responsible use of AI in a health-related project.

---

## Dataset

The project uses the:

**malaria_ssa_baseline_1000.csv**

dataset from the Electric Sheep Africa synthetic malaria dataset collection.

The dataset contains:

- 1,000 synthetic patient records.
- 28 original variables.
- Malaria-positive and malaria-negative cases.

The target variable used in this project is:

`malaria_status`

The target distribution was:

- Positive: 531
- Negative: 469

Because the dataset is synthetic, model performance should not be interpreted as real-world clinical performance.

---

## Features Used

The original dataset contained 28 variables.

To reduce data leakage, only information that could reasonably be available before laboratory diagnosis was selected.

The final model uses:

- Age
- Sex
- Residence
- Season
- Mosquito-net usage
- Fever duration
- Fever
- Chills
- Headache
- Vomiting
- Diarrhea
- Weakness / fatigue

The target variable is:

- Malaria status

Clinical or post-diagnostic variables such as parasite count, parasite species, malaria severity, and patient outcome were excluded from the model.

---

## Machine Learning Workflow

The project followed the following workflow:

1. Data collection
2. Data inspection
3. Missing-value analysis
4. Duplicate checking
5. Exploratory data analysis
6. Feature selection
7. Target encoding
8. Train-test splitting
9. Data preprocessing
10. Logistic Regression training
11. Random Forest training
12. Model evaluation
13. Five-fold cross-validation
14. Final model selection
15. Model persistence using Joblib
16. Streamlit application development

---

## Models Evaluated

Two classification algorithms were evaluated:

### Logistic Regression

Logistic Regression was used as a simple and interpretable baseline classification model.

### Random Forest

Random Forest was used as an ensemble machine-learning model capable of learning more complex relationships between features.

---

## Model Results

| Model | Test Accuracy | CV Accuracy | CV Precision | CV Recall | CV F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 100.0% | 99.90% | 100.0% | 99.81% | 99.91% |
| Random Forest | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

Random Forest was selected as the final model because it achieved the highest cross-validation performance.

However, the unusually high performance is largely related to the strongly separated patterns present in the synthetic dataset.

These results must therefore not be interpreted as evidence of real-world clinical accuracy.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Five-fold cross-validation

The final Random Forest model correctly classified all 200 records in the held-out test set.

---

## Application Architecture

The application combines machine learning with a rule-based safety layer.

```text
Patient Information
        |
        v
Reported Symptoms
        |
        v
Danger Sign Check
      /     \
    Yes      No
     |        |
     v        v
Urgent     Random Forest
Medical       |
Advice        v
          ML Prediction
              |
              v
        Triage Guidance
