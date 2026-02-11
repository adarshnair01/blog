---
layout: page
title: Media Optimization
description: Developed media optimization strategies to maximize Return on Investment (ROI) across various advertising channels.
img: assets/img/media-optimization.png
importance: 9
category: work
related_publications: false
---

# Algorithmic Media Mix Optimization

## Executive Summary

Marketing budgets are finite; the opportunities to spend them are infinite. This project focused on the _prescriptive_ side of marketing analytics: "Now that we have a model, how should we spend our $10M budget?" We developed a Media Optimization engine using non-linear optimization algorithms (BFGS, SLSQP) to finding the optimal allocation of budget across TV, Digital, Search, and Social channels to maximize Revenue/ROI.

## Problem Statement

The client allocated budgets based on "last year's spend plus 5%." This ignored diminishing returns (saturation) and cross-channel synergies. They were overspending on saturated channels (where the next dollar brings $0 return) and underspending on high-potential channels.

## Methodology

### 1. Response Curves (The Foundation)

You cannot optimize what you cannot model. We first built **Market Mix Models (MMM)** to generate "Response Curves" for each channel.

- **Shape**: Typically S-curves (Sigmoid) or Hill functions.
- **Diminishing Returns**: Captures the fact that the first $1k in ads works better than the millionth $1k.
- **Equation**: `Sales = β * Hill(Spend, K, S)` where K is half-saturation point and S is slope.

### 2. The Optimization Problem

- **Objective Function**: Maximize `Sum(Sales_channel_i)`
- **Constraints**:
  - `Total Spend <= Budget`
  - `Min_Spend_i <= Spend_i <= Max_Spend_i` (Business constraints, e.g., "We must spend at least $50k on Brand Search").
- **Algorithm**: We utilized `scipy.optimize` with the **L-BFGS-B** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno Boxed) algorithm, which is highly efficient for bound-constrained non-linear optimization problems.

### 3. Scenario Planner

We built a web-based "Simulator" tool.

- **Input**: "What if I have $5M? What if I have $10M?"
- **Output**: The tool visualizes the optimal split. It also shows the "Opportunity Cost" (how much revenue you lose by deviating from the optimal).

## Implementation Details

- **Tech Stack**: Python, Pandas, SciPy, Streamlit (for the UI).
- **Scalability**: Validated the optimizer on up to 50 distinct channels/geographies.
- **Integration**: Outputs were exported to Excel/Tableau for the Media Buying teams to execute.

## Challenges & Solutions

- **Challenge**: Local Optima. The curve might be complex.
- **Solution**: "Basinhopping" (Global Optimization). We ran the optimizer from multiple random starting points to ensure we found the global maximum, not just a local peak.

- **Challenge**: "The model says spend $0 on TV." (Political Challenge).
- **Solution**: Hard Constraints. We added capability for stakeholders to lock specific channels ("Force TV spend = $1M"), and the optimizer would find the _best possible_ solution given that constraint, while highlighting the efficiency loss.

## Results and Impact

- **ROI Improvement**: The optimized allocation was projected to deliver 15-20% more revenue for the _same_ total budget compared to the historical allocation.
- **Adoption**: The tool became the standard quarterly planning instrument for the CMO office.
- **Efficiency**: Automated a process that previously took 2 weeks of Excel work into a 5-minute run.

## Future Work

- **Multi-Objective Optimization**: Optimizing for _both_ Short-term Revenue AND Long-term Brand Equity (Awareness) simultaneously (Pareto Frontier).
- **Granularity**: Moving from channel-level (TV vs Digital) to campaign-level optimization.
