---
layout: page
title: Customer Lifetime Value Prediction
description: Predictive modeling to estimate the total value of customers over the entire relationship.
img: assets/img/cltv.png
importance: 6
category: work
related_publications: false
---

# Predicting Future Value: CLTV Modeling

## Executive Summary

Understanding the future value of a customer is crucial for setting Customer Acquisition Cost (CAC) targets. If you know a user will spend $500 over their lifetime, you can afford to spend $50 to acquire them. This project implemented a robust Customer Lifetime Value (CLTV) prediction model using the **Pareto-NBD / BG-NBD** probabilistic frameworks, allowing the finance and marketing teams to optimize spend with confidence.

## Problem Statement

The company was using "Historic CLTV" (sum of past purchases) as a proxy for value. This is backward-looking and undervalues new users who might have high potential but haven't spent much *yet*. We needed a *predictive* measure: "How much will this user value vary in the next 12 months?"

## Methodology

### 1. "Buy 'Til You Die" (BTYD) Models
We utilized the **Lifetimes** library in Python to implement probabilistic models:
   - **BG/NBD (Beta-Geometric / Negative Binomial Distribution)**: Models the transaction process (frequency and recency). It assumes users buy at a constant rate until they "die" (churn).
   - **Gamma-Gamma Model**: Models the monetary value (average order value). It assumes the monetary value of a customer's transactions varies randomly around their average transaction value.

### 2. Feature Data
   - `frequency`: Number of repeat purchases.
   - `recency`: Age of the customer when they made their last purchase.
   - `T`: Age of the customer (time since first purchase).
   - `monetary_value`: Average value of a customer's purchases.

### 3. Validation
   - **Calibration vs. Holdout**: We split the data into a calibration period (e.g., 2023) and a holdout period (e.g., 2024). We trained on 2023 data and predicted 2024 transactions.
   - **Metrics**: RMSE, MAE between predicted and actual revenue in the holdout period.

## Implementation Details

-   **Batch Processing**: The model is retrained monthly on the updated transaction history.
-   **Scalability**: While BTYD models are efficient, for very large datasets (10M+ rows), we utilized distributed implementations or sampled datasets.
-   **Dashboarding**: A Tableau dashboard visualizes the "Predicted CLTV" distribution, helping execs see the health of the customer base (e.g., "Are we acquiring higher quality users this year vs last?").

## Use Cases Enabled

1.  **CAC Optimization**: Marketing can bid higher for "High LTV" lookalike audiences on Facebook.
2.  **Resource Allocation**: Customer Success teams prioritize tickets from "High Future Value" clients.
3.  **Financial Planning**: Finance uses the aggregated predicted revenue for quarterly forecasting.

## Challenges & Solutions

-   **Challenge**: The model struggled with one-time operational anomalies (e.g., a massive clearance sale that spiked frequency artificially).
-   **Solution**: Implemented cohort-based adjustments and outlier removal for anomalous periods to stabilize model parameters.

## Results and Impact

-   **Accuracy**: The BG/NBD model predicted aggregate revenue within 5% of actuals for the 6-month holdout period.
-   **Marketing ROI**: By shifting budget to acquire "High Predicted LTV" users, the 6-month payback period improved by 20%.

## Future Work

-   **Deep Learning (DNN)**: Experimenting with deep neural networks (using features beyond just RFM, like clickstream data) to see if they outperform the probabilistic BTYD models for individual-level prediction accuracy.
