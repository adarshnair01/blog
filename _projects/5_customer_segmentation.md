---
layout: page
title: Customer Segmentation
description: Advanced customer segmentation solution to tailor marketing strategies to specific user groups.
img: assets/img/customer_segmentation.png
importance: 5
category: work
related_publications: false
---

# Feature-Rich Customer Segmentation

## Executive Summary

One-size-fits-all marketing is dead. To stay relevant, businesses must understand the diverse personas within their customer base. This project delivered a comprehensive Customer Segmentation engine using unsupervised learning (Clustering). It identified 6 distinct customer personas (e.g., "Bargain Hunters," "Loyal Enthusiasts," "Dormant High-Value"), enabling the marketing team to tailor messaging, offers, and product recommendations for each group, resulting in a 14% increase in email engagement.

## Problem Statement

The business relied on basic heuristic segmentation (e.g., "Bought in last 30 days"). This failed to capture behavioral nuances—a customer buying a single expensive item once is very different from one buying cheap items weekly, yet they might look similar in total spend. We needed a multi-dimensional segmentation approach.

## Methodology

### 1. Feature Selection (RFM+)
   - **RFM**: Recency, Frequency, Monetary Value.
   - **Behavioral**: Average time on site, preferred categories, discount sensitivity (ratio of purchases made on sale), return rate.
   - **Demographic**: Location tier, age group.

### 2. Preprocessing
   - **Scaling**: Standardized features using `StandardScaler` (Z-score normalization) to ensure features with large ranges (e.g., Revenue) didn't dominate features with small ranges (e.g., Frequency).
   - **transformation**: Applied log transformation to skewed distributions (like Spend).

### 3. Clustering Approach
   - **K-Means**: Selected for its efficiency and interpretability.
   - **Elbow Method**: Used to determine the optimal number of clusters (k=6).
   - **Silhouette Score**: Validated cluster quality/separation.

## Implementation Details

The segments are recalculated weekly to account for changing user behavior.

-   **Pipeline**: Python (Pandas + Scikit-Learn) script running on AWS Lambda (triggered by EventBridge).
-   **Output**: A `segment_id` and `segment_name` tag is applied to each user profile in the Data Warehouse (Snowflake).
-   **Integration**: These tags are synced to the Email Service Provider (ESP) and Push Notification tools appropriately.

### The 6 Personas identified:
1.  **Champions**: High spend, high frequency, recent. (Strategy: VIP rewards, early access).
2.  **Loyal Potential**: High frequency, low spend. (Strategy: Upsell/Cross-sell).
3.  **Big Spenders**: Low frequency, high spend. (Strategy: Nurture with premium content).
4.  **Promiscuous**: Only buy on deep discount. (Strategy: Flash sales, clearance).
5.  **At Risk**: Previously good customers, high recency. (Strategy: Win-back campaigns).
6.  **Hibernate**: Low value, haven't visited in long time. (Strategy: Low-cost re-engagement).

## Challenges & Solutions

-   **Challenge**: Interpretability. "What does Cluster 3 mean?"
-   **Solution**: Created "Snake Plots" and "Relative Importance Heatmaps" to visualize how each cluster differs from the population average on every feature.

-   **Challenge**: Stability. Customers "jumping" between segments too frequently.
-   **Solution**: Implemented a smoothing logic where a user must exhibit behaviors of a new segment for 2 consecutive weeks before being moved, preventing marketing whiplash.

## Results and Impact

-   **Engagement**: Customized subject lines for "Bargain Hunters" vs "Champions" led to a 14% increase in Open Rates.
-   **Revenue**: "At Risk" win-back campaigns recovered $50k/month in potentially lost revenue.
-   **Strategy**: This segmentation is now the "common language" used across Product, Marketing, and Sales teams to discuss user groups.

## Future Work

-   **Micro-Segmentation**: Breaking down these 6 macro-segments into smaller niches (e.g., "At Risk" -> "At Risk - High Value" vs "At Risk - Low Value").
-   **Persona Evolution**: Tracking how users move between segments over their lifecycle (Markov Chains) to predict LTV trajectories.
