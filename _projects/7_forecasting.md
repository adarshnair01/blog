---
layout: page
title: Forecasting for Pharma Client (SKUs)
description: End-to-end sales forecasting system for inventory planning in the pharmaceutical sector.
img: assets/img/pharma-forecasting.jpg
importance: 7
category: work
related_publications: false
---

# Precision SKU Forecasting for Pharmaceuticals

## Executive Summary

Supply chain efficiency in pharmaceuticals is a matter of critical cost and patient safety. Understocking leads to lost sales and patient risk; overstocking leads to expiry and waste. This project delivered a highly granular Sales Forecasting System for a major pharma client, predicting sales for over 2,000 SKUs across 50 regions. The solution reduced forecast error (MAPE) by 18%, directly translating to millions in inventory savings.

## Problem Statement

The client was using simple moving averages (Excel) for forecasting. This failed to account for:

- Seasonality (flu season, allergy season).
- Trends (growing/shrinking markets).
- Exogenous factors (competitor price changes, regulatory approvals).
- Hierarchy (national trends vs regional variances).

## Methodology

### 1. Hierarchical Time Series Forecasting

We needed to ensure that the sum of regional forecasts equaled the national forecast (reconciliation).

- **Bottom-Up Approach**: Forecasting at the lowest level (SKU-Region) and aggregating.
- **Top-Down Approach**: Forecasting at the National level and disaggregating based on historical proportions.
- **Optimal Reconciliation (MinT)**: We eventually used the MinT (Minimum Trace) reconciliation method to optimally combine forecasts at all levels.

### 2. Models Evaluated

- **ARIMA / SARIMA**: Strong baseline for univariate series with clear seasonality.
- **Prophet (Facebook)**: Excellent for handling holidays and changepoints (e.g., new regulation introduced).
- **XGBoost (Regression)**: Treated forecasting as a supervised regression problem, using lagged values as features.
- **Ensemble**: The final solution was a weighted _ensemble_ of Prophet and XGBoost.

### 3. Feature Engineering

- **Lag Features**: Sales(t-1), Sales(t-12), etc.
- **Calendar Events**: `is_holiday`, `month_of_year`, `flu_season_index`.
- **External Data**: FDA approval dates, competitor pricing index.

## Implementation Details

- **Pipeline**: Python ecosystem (`statsmodels`, `prophet`, `scikit-learn`).
- **Orchestration**: Airflow DAGs managed the weekly retraining and inference jobs.
- **Forecast Horizon**: Weekly forecasts for the next 12 months (52 weeks).
- **Output**: A clean CSV feed into the client's ERP system (SAP) for automated replenishment planning.

## Challenges & Solutions

- **Challenge**: "New Product Launch". How to forecast an SKU with no history?
- **Solution**: Implemented "Reference Class Forecasting". We clustered historical products to find "similar" past launches and used their launch curves as a proxy for the new product.

- **Challenge**: Sparse Data. Some SKUs sold very intermittently (lots of zeros).
- **Solution**: Used Croston’s method (forecasting _interval_ between sales and _size_ of sale separately) for intermittent and lumpy demand patterns.

## Results and Impact

- **Accuracy**: Reduced Weighted MAPE (Mean Absolute Percentage Error) from 35% (Excel) to 17% (Ensemble Model).
- **Inventory**: Enabled a 12% reduction in safety stock levels while maintaining service levels (fill rate).
- **Waste**: Projected reduction in expired inventory by ~8% annually.

## Future Work

- **Demand Sensing**: Incorporating real-time downstream data (POS data from pharmacies) to adjust forecasts daily rather than weekly.
- **Causal AI**: Using Causal Inference to better understand the _impact_ of pricing changes ("If we raise price by 5%, how much will volume drop?") rather than just correlating them.
