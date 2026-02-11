---
layout: page
title: Fine-tuning LLM for Intents
description: Custom Fine-Tuned Large Language Models for specific domain intent recognition.
img: assets/img/genai-chatbot.jpg
importance: 1
category: work
related_publications: false
---

# Fine-tuning LLM for Intent Recognition

## Executive Summary

The widespread adoption of Large Language Models (LLMs) has revolutionized natural language processing, yet generic models often struggle with highly specialized domain language and specific intent recognition. This project aimed to bridge that gap by fine-tuning open-source LLMs (specifically LLaMA 3 and Mistral 7B) on a proprietary dataset of customer interactions. The result was a specialized model capable of identifying over 150 distinct user intents with an accuracy exceeding 94%, significantly outperforming generic zero-shot baselines and reducing latency by 40% compared to using larger, api-based models.

## Problem Statement

In a customer support environment handling thousands of queries daily, accurately routing requests is critical for efficiency. Traditional intent classifiers (using BERT or simpler ML models) often failed to capture nuance or handle complex, multi-intent queries. Generic LLMs were too slow and costly for real-time classification, and often "hallucinated" categories not in our taxonomy. We needed a solution that combined the reasoning capabilities of an LLM with the strict adherence to a defined intent schema required for enterprise routing.

## Methodology

### 1. Data Curation and Synthesis

- **Historical Data**: Aggregated 500,000+ historical chat logs, manually labeled by support agents.
- **Synthetic Generation**: Used GPT-4 to generate synthetic variations of rare intents to balance the dataset, employing "chain-of-thought" prompting to ensure diversity in phrasing while maintaining semantic integrity.
- **Cleaning**: Implemented rigorous deduplication and PII redaction pipelines using scrubadub and custom regex filters.

### 2. Model Selection and Architecture

- **Base Models**: Evaluated LLaMA 3 (8B) and Mistral (7B) for their balance of performance and inference speed.
- **LoRA (Low-Rank Adaptation)**: Utilized LoRA to fine-tune only a small subset of parameters (approx. 1-2%), significantly reducing computational requirements while maintaining the base model's general linguistic capabilities.
- **Config**:
  - Rank (r): 64
  - Alpha: 16
  - Dropout: 0.1
  - Quantization: 4-bit (QLoRA) for training on single A100 GPUs.

### 3. Training Process

- **Instruction Tuning**: Formatted data into `(System, User, Assistant)` tuples, where the system prompt defined the taxonomy and the assistant output was the JSON-structured intent.
- **Hyperparameters**:
  - Learning rate: 2e-4
  - Batch size: 128 (with accumulation)
  - Epochs: 3 (early stopping based on validation loss)
- **Framework**: Utilized the `Axolotl` library and `Unsloth` for optimized training speeds (2x faster than standard HuggingFace Trainer).

## Implementation Details

The fine-tuning pipeline was containerized using Docker and orchestrated via Kubernetes (EKS).

- **Inference Server**: Deployed the fine-tuned model using `vLLM` (Versatile Large Language Model) serving engine, which utilizes PagedAttention to maximize throughput.
- **Quantization**: The final model was served in 4-bit quantization using AWQ (Activation-aware Weight Quantization) to fit within 24GB VRAM consumer-grade GPUs for cost efficiency.

```python
# Sample Inference Code using vLLM
from vllm import LLM, SamplingParams

llm = LLM(model="models/llama-3-8b-intent-finetuned-v1", quantization="awq")
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

outputs = llm.generate(prompts, sampling_params)
```

## Results and Impact

- **Accuracy**: Achieved 94.2% F1-score on the held-out test set, a 12% improvement over the previous BERT-based baseline.
- **Latency**: Average latency per request was ~150ms on an A10G GPU, making it suitable for real-time chat applications.
- **Cost**: Reduced operational costs by 70% compared to using GPT-3.5-Turbo API calls for the same volume of requests.

## Future Work

Future iterations will focus on:

- **Direct Preference Optimization (DPO)**: Further aligning the model's outputs with human preference data to reduce hallucinations.
- **Continuous Learning**: Implementing a feedback loop where corrected predictions from human agents are automatically added to the training dataset for weekly model updates.
