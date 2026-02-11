---
layout: page
title: Market Mix Modelling
description: Built comprehensive market mix modeling capabilities to quantify the impact of marketing and non-marketing drivers on sales.
img: assets/img/mmm.png
importance: 10
category: work
related_publications: false
---

# Bayesian Market Mix Modeling (MMM)

## Executive Summary

"Half the money I spend on advertising is wasted; the trouble is I don't know which half." - John Wanamaker.
This project aimed to solve the age-old marketing attribution problem. We built a robust Market Mix Modeling (MMM) solution using Bayesian priors to quantify the incremental impact of each marketing channel (TV, FB, Radio, etc.) on sales, accounting for lag effects (Adstock) and saturation.

## Problem Statement

Multi-Touch Attribution (MTA) (tracking individual user clicks) was becoming impossible due to privacy changes (iOS14, cookie deprecation) and "Walled Gardens" (Facebook/Google not sharing data). The client needed a **privacy-first**, top-down statistical approach to understand marketing effectiveness without needing user-level tracking.

## Methodology

### 1. Data Collection
   - **Dependent Variable**: Weekly Sales (Revenue or Units).
   - **Independent Variables (Media)**: Weekly Spend and Impressions by channel.
   - **Controls**: Price, Distribution (ACV), Competitor activity, Macroeconomic indicators (GDP, Unemployment), Seasonality, Holidays.

### 2. Bayesian Approach (LightweightMMM / Robyn)
   - Unlike frequentist OLS regression, a Bayesian approach allows us to inject **Prior Knowledge**.
   - **Example**: "We know from lift tests that Facebook ROI is likely between 2.0 and 4.0." We set priors on the coefficients to constrain the model to reality, preventing nonsensical results (like negative impact of ads).
   - We used **PyMC3** / **NumPyro** for MCMC sampling.

### 3. Transformations
   - **Adstock (Carryover)**: Ads seen today affect sales next week. We modeled this geometric decay.
   - **Saturation (Diminishing Returns)**: Modeled using Hill functions.

## Implementation Details

-   **Validation**:
    -   **MAPE**: In-sample and Out-of-sample error.
    -   **Lift Tests**: We calibrated the model using results from Geo-Lift experiments (e.g., "We turned off TV in Ohio for 2 weeks, sales dropped 5%"). This "Ground Truth" improved model trust significantly.

## Output & Insights

1.  **Contribution Charts**: "Waterfalls" showing that 40% of sales comes from Baseline (Brand Equity), 15% from TV, 10% from Search, etc.
2.  **ROI / ROAS**: Calculated the Return on Ad Spend for each channel.
3.  **Marginal ROI**: The most critical metric. "If I spend the *next* dollar, how much will I get?" (mROAS is often lower than Average ROAS due to saturation).

## Challenges & Solutions

-   **Challenge**: Collinearity. TV spend and Search spend often go up and down together. Regression struggles to separate them.
-   **Solution**: Ridge Regression (L2 Regularization) and strong informative Priors helped separate the effects.

## Results and Impact

-   **Budget Shift**: Identified that Display Ads were severely non-performing (ROI < 0.5) while Brand Search was undersaturated.
-   **Savings**: Reallocated $2M annual budget based on model findings, driving an estimated $500k incremental profit.
-   **Production**: Established a quarterly cadence for model updates.

## Future Work

-   **Hierarchical MMM**: Modeling national and regional data simultaneously (Geo-level modeling) to increase data points (`N_regions * T_weeks`) and reduce collinearity.
-   **Automated Pipeline**: Moving from a consulting-style manual build to an automated "Always-on" MMM platform.
