---
layout: page
title: WhatsApp Chatbot using LLMs
description: AI-driven customer support chatbot integrated with WhatsApp using Large Language Models.
img: assets/img/whatsapp-bot.jpeg
importance: 2
category: work
related_publications: false
---

# Enterprise WhatsApp Chatbot with RAG and LLMs

## Executive Summary

Customer expectations for instant, accurate support are at an all-time high. To meet this demand, we developed an intelligent WhatsApp chatbot capable of handling complex customer queries autonomously. By integrating Large Language Models (LLMs) with a Retrieval-Augmented Generation (RAG) architecture, the bot can access up-to-date company policies, product manuals, and user data to provide personalized and accurate responses, deflecting over 60% of routine tickets from human agents.

## Problem Statement

Our previous rule-based chatbot was rigid and frustrating for users. It could only handle exact keyword matches and frequently trapped users in loops ("I typically reply in 5 minutes"). Humans agents were overwhelmed with repetitive queries about order status, return policies, and product details. We needed a solution that could understand natural language, maintain context over a conversation, and fetch real-time data.

## Methodology

### 1. Architecture: RAG Pipeline
   - **Vector Database**: We used Weaviate to index over 5,000 pages of internal documentation, FAQs, and product manuals.
   - **Embeddings**: Utilized OpenAI's `text-embedding-3-small` for high-quality, cost-effective semantic search.
   - **Retrieval**: Implementation of a hybrid search strategy (Sparse BM25 + Dense Semantic Vector) to ensure keywords (like specific error codes) and semantic concepts were both captured.

### 2. LLM Integration
   - **Orchestration**: Built using **LangChain** to manage the conversation flow, memory, and tool usage.
   - **Model**: GPT-4-Turbo was used for the reasoning engine due to its superior instruction following and reduced hallucination rates compared to smaller models.
   - **Tools**: The LLM was equipped with "tools" (function calling) to:
     - Check Order Status (API call to Shopify/OMS)
     - Validate User Identity (OTP flow)
     - Escalate to Human (Intercom API integration)

### 3. WhatsApp Integration
   - **BSP**: Utilized Twilio as the Business Service Provider (BSP) for the WhatsApp Business API.
   - **Webhook Handler**: Developed a robust FastAPI webhook handler to receive messages, process them asynchronously via Celery workers, and send responses.

## Implementation Details

The core of the system is the "Agentic" workflow. Instead of a linear script, the LLM acts as an agent that decides the next step.

**Tech Stack:**
- **Backend**: Python, FastAPI, Celery
- **LLM Ops**: LangChain, LangSmith (for tracing)
- **Database**: PostgreSQL (User Session State), Weaviate (Knowledge Base)
- **Messaging**: Twilio WhatsApp API

**Context Management:**
To handle the stateless nature of HTTP webhooks, we implemented a Redis-backed session manager that retrieves the last `k` turns of conversation for every incoming message, allowing the user to say "What about the second one?" and have the bot understand they are referring to the second product listed in the previous message.

## Challenges & Solutions

- **Challenge**: WhatsApp enforces a strict 24-hour customer service window.
- **Solution**: Implemented a template message re-engagement flow. If a user returns after 24 hours, the bot automatically triggers an approved template message to restart the session.

- **Challenge**: The LLM would sometimes make up policies (Hallucination).
- **Solution**: Implemented strict "Grounding" prompts. "Answer solely based on the retrieved context below. If the answer is not in the context, say 'I'm not sure, let me connect you with a human.'"

## Results

- **Deflection Rate**: Successfully automated 65% of incoming L1 support queries.
- **CSAT**: Verified Customer Satisfaction Score (CSAT) increased from 3.8/5 to 4.6/5.
- **Response Time**: Reduced median first response time from 45 minutes (human) to <5 seconds (bot).

## Future Work

- **Multimodal Support**: Adding the ability for users to send images (e.g., of a damaged product) which the bot can analyze using GPT-4 Vision.
- **Voice Support**: Allowing users to send voice notes which are transcribed (Whisper) and processed textually.
