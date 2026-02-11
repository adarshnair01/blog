---
layout: page
title: Voice of Consumer
description: NLP based social listening tool to analyze consumer sentiment and feedback.
img: assets/img/voc.png
importance: 8
category: work
related_publications: false
---

# Voice of Consumer (VoC) Analytics Engine

## Executive Summary

Brands often struggle to hear the signal in the noise of social media. This project involved building a scalable NLP-powered "Voice of Consumer" engine. It ingested data from Twitter, Reddit, and Customer Support tickets to provide real-time dashboards on Brand Sentiment, Emerging Topics, and Crisis Detection. The platform processed over 3 million mentions daily with 99.95% uptime.

## Problem Statement

The client, a large consumer brand, relied on manual reading of comments and basic keyword searches to gauge public opinion. This was:

1.  **Reactive**: They only found out about issues after they blew up.
2.  **Biased**: Manual reviewers only saw a tiny sample.
3.  **Unstructured**: No quantitative metrics to track improvement.

## Methodology

### 1. Data Ingestion

- **Sources**: Twitter Firehose API, Reddit API (PRAW), internal ZenDesk dumps.
- **Streaming**: Used **Apache Kafka** to handle the high-throughput ingestion of widely varying text formats.

### 2. Natural Language Processing (NLP) Pipeline

- **Preprocessing**: Cleaning, tokenization, stop-word removal, and normalization (handling emojis/slang).
- **Sentiment Analysis**: Fine-tuned a **RoBERTa** model on domain-specific tweets. (Generic models failed on sarcasm and brand-specific context).
- **Topic Modeling**: Utilized **BERTopic** (and previously LDA) to unsupervisedly discover "what people are talking about" (e.g., "Packaging complaints", "Flavor profile", "Shipping delay").
- **Named Entity Recognition (NER)**: Extracting competitor names, locations, and product SKUs.

### 3. Latency Optimization

- **Challenge**: The Transformer models were slow (300ms+ per tweet).
- **Solution**:
  - **Distillation**: Switched to DistilRoBERTa.
  - **ONNX Runtime**: Converted models to ONNX format for accelerated inference on CPU.
  - **Result**: Reduced inference latency from 320ms to 120ms (62% reduction).

## Implementation Details

- **Backend**: Python (FastAPI) services.
- **Storage**:
  - **Elasticsearch**: For full-text search and aggregations (e.g., "Show me negative tweets about 'vanilla' in 'New York'").
  - **S3**: For data lakes / long-term archival.
- **Frontend**: A React-based command center where Brand Managers could see live feeds and sentiment trendlines.

## Challenges & Solutions

- **Challenge**: Multilingual data.
- **Solution**: Implemented a language detection layer (fastText). Non-English tweets were routed to specific translation APIs (Google Translate) or language-specific models (XLM-RoBERTa).

- **Challenge**: Spam / Bots.
- **Solution**: Trained a lightweight Logistic Regression classifier on metadata (account age, post frequency) to flag and filter out bot activity before it skewed sentiment metrics.

## Results and Impact

- **Crisis Management**: Successfully alerted the PR team to a potential PR crisis (a viral complaint video) 2 hours before it trended nationally, allowing for a proactive response.
- **Product Insights**: Identified a recurring complaint about "packaging seals breaking" which led to a supply chain audit and package redesign.
- **Client Satisfaction**: Boosted client satisfaction scores by 12% by providing them with actionable, data-backed insights rather than anecdotes.

## Future Work

- **Aspect-Based Sentiment Analysis (ABSA)**: Moving beyond "This tweet is negative" to "This tweet is negative regarding _Price_ but positive regarding _Quality_."
- **Image Analysis**: Analyzing images (memes, product photos) in posts to capture visual sentiment.
