---
layout: page
title: Propensity Score for Customers
description: Propensity modeling framework to predict customer likelihood to purchase or churn.
img: assets/img/propensity-modeling.jpg
importance: 4
category: work
related_publications: false
---

# Propensity Modeling Framework: Predicting Customer Actions

## Executive Summary

Marketing efficiency hinges on targeting the right customer at the right time. This project developed a suite of propensity models to predict the likelihood of key customer actions: purchasing (Conversion Propensity) and churning (Churn Propensity). By integrating these scores into our CRM and ad platforms, we achieved a 22% increase in campaign ROI and a 15% reduction in churn rate for high-risk segments.

## Problem Statement

Our marketing campaigns were previously broad and unoptimized, often targeting users who were unlikely to buy (wasting ad spend) or ignoring high-risk customers who were about to leave. We needed a data-driven way to score _every_ customer daily on their probability to take specific actions in the next 7-30 days.

## Methodology

### 1. Data Pipeline & Feature Engineering

- **Behavioral Data**: aggregated clickstream events (page views, cart adds, search queries) from Google Analytics/Segment.
- **Transactional Data**: Purchase history, Average Order Value (AOV), recency of purchase.
- **Demographic Data**: Age, location, device type.
- **Engineered Features**:
  - `days_since_last_active`
  - `session_duration_avg`
  - `cart_abandonment_rate`
  - `category_affinity_score`

### 2. Modeling Approach

- **Algorithm**: XGBoost (Extreme Gradient Boosting) was chosen for its performance on tabular data and interpretability (via SHAP values).
- **Target Variable**: Binary classification (1 = Event occurred in window, 0 = Did not).
- **Handling Imbalance**: Utilized SMOTE (Synthetic Minority Over-sampling Technique) and `scale_pos_weight` to handle the class imbalance (conversion rates are typically low, <5%).
- **Validation**: Time-series cross-validation (training on past months, validating on future months) to prevent data leakage.

### 3. Model Explainability

- **SHAP (SHapley Additive exPlanations)**: Used to explain _why_ a specific customer had a high score. e.g., "High likelihood to churn because `days_since_last_order` > 90 and `support_ticket_sentiment` is negative."
- These insights were pushed to the CRM for agents to see.

## Implementation Details

The system runs as a daily batch job on Databricks/Spark.

1.  **ETL**: Nightly job aggregates data from the Data Lake (S3/Delta Lake).
2.  **Inference**: The trained XGBoost model scores all 5M+ active users.
3.  **Activation**:
    - **High Propensity (Purchase)** -> Pushed to Facebook/Google Ads as "High Intent Audience".
    - **High Propensity (Churn)** -> Pushed to Salesforce/Email tool for a "We Miss You" retention campaign.
    - **Low Propensity** -> Excluded from expensive campaigns to save budget.

## Challenges & Solutions

- **Challenge**: "Why did my score drop?" Stakeholders needed transparency.
- **Solution**: Built a Streamlit dashboard allowing marketers to input a Customer ID and see the top 5 features contributing to their score (positive/negative).

- **Challenge**: Model drift over time (e.g., during Black Friday).
- **Solution**: Implemented automated retraining pipelines using Airflow that trigger if model performance metrics (AUC-ROC) drop below a threshold on the previous day's data.

## Results and Impact

- **Conversion Rate**: Email campaigns targeting the top 20% propensity decile saw a 3x higher open rate and 2x higher conversion rate.
- **Ad Spend Efficiency**: Reduced Cost Per Acquisition (CPA) by 18% by suppressing ads to low-propensity users.
- **Retention**: The proactive churn prevention campaign saved ~1,200 high-value customers per month.

## Future Work

- **Uplift Modeling**: Instead of just predicting _who_ will buy, predicting _who is persuadable_ (i.e., would only buy _if_ we show them an ad).
- **Real-time Scoring**: Moving from batch to real-time scoring using Feature Stores (Feast) to react to user actions within seconds.
