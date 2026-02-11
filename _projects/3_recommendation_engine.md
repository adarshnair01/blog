---
layout: page
title: Recommendation Engine (2-Tower TFRS)
description: Personalized recommendation system using TensorFlow Recommenders and Two-Tower architecture.
img: assets/img/recommender-system.png
importance: 3
category: work
related_publications: false
---

# Scalable Two-Tower Recommendation Engine

## Executive Summary

In the competitive landscape of e-commerce, personalization is the key driver of conversion. This project involved designing and deploying a state-of-the-art recommendation system using the **Two-Tower architecture** within the **TensorFlow Recommenders (TFRS)** framework. The system was designed to handle millions of users and items, delivering personalized product suggestions in real-time (under 50ms latency), resulting in a 15% uplift in Click-Through Rate (CTR) and a 9% increase in Gross Merchandise Value (GMV).

## Problem Statement

Our legacy recommendation system relied on classical Collaborative Filtering (Matrix Factorization) which suffered from the "cold start" problem and couldn't scale efficiently with our rapidly growing product catalog. It also failed to utilize rich content features (product descriptions, images) and context (time of day, device type). We needed a deep learning-based solution that could leverage these auxiliary features and scale linearly.

## Methodology

### 1. Two-Tower Architecture

The core concept involves two separate neural networks (towers):

- **Query Tower (User Tower)**: Encodes user features (ID, demographics, past history) into a fixed-size embedding vector.
- **Candidate Tower (Item Tower)**: Encodes item features (ID, category, description embeddings) into a fixed-size embedding vector of the same dimension.
- **Dot Product**: The similarity score (prediction) is calculated as the dot product of the Question and Candidate vectors. This allows for extremely fast retrieval using Approximate Nearest Neighbor (ANN) search.

### 2. Feature Engineering

- **User Features**:
  - `user_id` (Embedding)
  - `history_last_10_items` (Sequence embedding, averaged or processed via LSTM)
  - `geolocation` (Hashed)
- **Item Features**:
  - `item_id` (Embedding)
  - `title` (Text embedding via DistilBERT)
  - `category` (Categorical embedding)
  - `price_normalized` (Continuous)

### 3. Model Training

- **Framework**: TensorFlow Recommenders (TFRS)
- **Objective**: Retrieval task (Top-K) using a Softmax/Logits loss.
- **Negative Sampling**: Implemented "in-batch" negative sampling for efficient training on large datasets without explicit negative pair generation.
- **Optimizer**: Adagrad with a dynamic learning rate schedule.

## Implementation Details

The deployment pipeline is critical for a high-traffic recommendation system.

### Training Pipeline

- **Data**: TFRecord datasets stored in GCS.
- **Training**: Vertex AI (Google Cloud) Custom Training Jobs.
- **Embedding Generation**: Once trained, the Item Tower is used to pre-compute embeddings for all active products. These are exported to an ANN Index.

### Serving Infrastructure

- **ANN Index**: We used **ScaNN** (Scalable Nearest Neighbors) for the index service, hosted on Kubernetes.
- **Online Service**: A lightweight gRPC service computes the User Embedding on the fly (using the Query Tower) and queries the ScaNN index for the nearest product neighbors.
- **Ranking**: A second-stage "Ranking Model" (DLRM) re-ranks the top 100 candidates to optimize for conversion (CVR) rather than just relevance, taking into account real-time inventory and business logic.

## Challenges & Solutions

- **Challenge**: Latency spikes during high traffic.
- **Solution**: Implemented aggressive caching of User Embeddings for frequent visitors using Redis, reducing redundant computations.

- **Challenge**: The "Harry Potter" Effect (Popularity Bias). The model recommended popular items to everyone.
- **Solution**: Implemented "logQ" correction during training to penalize purely popular items and a post-processing dithering algorithm to inject diversity into the recommendations.

## Results and Impact

- **CTR Uplift**: +15.3% vs. the legacy MF baseline.
- **Conversion Rate**: +8.5% increase.
- **Catalog Coverage**: Improved catalog exposure by 40%, successfully surfacing "long-tail" products that were previously buried.
- **Latency**: Maintained a p99 latency of 48ms even at 10,000 QPS.

## Future Work

- **Real-time Session-based Recs**: Integrating Transformer4Rec to model user intent within a _single session_ in real-time.
- **Multi-Task Learning**: Optimizing for multiple objectives simultaneously (Clicks, Add-to-Cart, Purchase) using a shared-bottom, multi-tower wide & deep architecture.
