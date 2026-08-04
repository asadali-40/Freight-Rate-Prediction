# Freight Rate Prediction Challenge

## Overview

This repository contains my solution for the Freight Rate Machine Learning Assessment.

The objective of this project is to predict freight rates for unseen shipment records using historical freight transportation data. The solution includes data preprocessing, feature engineering, model training, evaluation, prediction generation, and validation using the provided scoring script.

---

## Project Structure

```
Freight-Rate-ML-Assessment
│
├── data
│   ├── train-test.csv
│   ├── validation.csv
│   ├── validation-predictions-template.csv
│   └── december-chart-inputs.csv
│
├── notebooks
│   └── Freight_Rate_Assessment.ipynb
│
├── scorer_results
│   └── candidate_december.png
│
├── validation_predictions.csv
├── december-chart-inputs.csv
├── score.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Dataset

The dataset contains freight shipment information including:

- Pickup location
- Delivery location
- Pickup & Delivery Coordinates
- Distance
- Equipment Type
- Weight
- Market Index
- Quote Signal
- Shipment Date
- Posted Rate (Target)

---

## Data Preprocessing

The following preprocessing steps were applied:

- Missing value imputation
- Date feature extraction
- Categorical feature handling
- Feature engineering
- Train / Validation split

Date features created:

- Month
- Day
- Day of Week
- Week of Year
- Weekend Indicator

---

## Model

After experimenting with different approaches, **CatBoost Regressor** was selected because it naturally handles categorical variables and delivered the best validation performance.

---

## Validation Performance

| Metric | Score |
|---------|------:|
| MAE | 108.85 |
| RMSE | 528.08 |
| R² Score | 0.8696 |

---

## Generated Files

The project generates:

- validation_predictions.csv
- december-chart-inputs.csv
- scorer_results/candidate_december.png

---

## Validation

Run the scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

Expected Output:

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
```

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Author

Machine Learning Assessment Submission

