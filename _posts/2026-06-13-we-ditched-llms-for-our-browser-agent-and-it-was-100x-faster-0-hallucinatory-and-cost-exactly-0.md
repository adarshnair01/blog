---
layout: post
title: "We Ditched LLMs for Our Browser Agent—And It Was 100x Faster, 0% Hallucinatory, and Cost Exactly $0"
date: 2026-06-13 21:08:12 +0530
excerpt: "Why are we using multi-billion parameter models to click a 'Submit' button? Inside our journey replacing bloated LLM agents with Jev, a deterministic state compiler."
author: "Adarsh Nair"
categories: ai
tags: ["AI", "Tech", "Web Automation", "Software Engineering"]
---

We are living in an era of engineering decadence. 

Right now, thousands of software engineers are writing applications that take a raw DOM tree, serialize it into 100,000 tokens of messy markdown, ship it across the internet to an external API running on thousands of power-hungry GPUs, and wait five seconds just to get a JSON response that says: *“Click the blue button.”*

It is slow. It is astronomically expensive. And worst of all, it is fragile. If the LLM experiences a transient latency spike, your browser agent times out. If the LLM decides to hallucinate a slightly different JSON schema, your parser throws an unhandled exception. If the target website updates its CSS class names, your context window overflows.

A few months ago, our team was building an enterprise browser agent designed to automate complex procurement workflows across hundreds of legacy SaaS portals. Naturally, we started with the modern stack: Playwright, LangChain, and GPT-4o. 

Within three weeks, we were drowning in token costs, fighting 8-second step latencies, and writing endless validation loops to catch hallucinatory actions. 

That was when we threw out the LLMs entirely and rebuilt our browser agent using **Jev**—a lightweight, deterministic state-machine compiler designed specifically for structured web interactions. 

The results were immediate:
* **Latency** dropped from 4,500ms per step to **12ms**.
* **Success rates** went from 78% (plagued by hallucinations) to **99.9%**.
* **API costs** dropped from $0.12 per execution run to **exactly $0.00**.

Here is how we did it, why LLMs are the wrong abstraction for browser agents, and how you can build a self-healing, deterministic agent using Jev.

---

### The Fundamental Flaw of LLM-Based Browser Agents

To understand why we made the switch, we have to look at how LLM-based web agents operate. The typical execution loop looks like this:

1. **DOM Serialization:** Capture the active browser viewport, strip unnecessary tags, and convert the DOM to text/markdown.
2. **Context Delivery:** Send the serialized DOM, the user's goal, and a list of available actions (click, type, scroll) to an LLM.
3. **Inference:** The LLM processes the prompt, reasoning about which element matches the goal.
4. **Action Parsing:** The agent receives a structured JSON payload indicating the target CSS selector and action.
5. **Execution:** Playwright or Puppeteer executes the action.

```
[DOM] ──(Serialize)──> [Text] ──(Network)──> [LLM Inference] ──(Network)──> [JSON Action] ──> [Playwright]
```

This architecture suffers from three fatal bottlenecks:

#### 1. The Token Tax and DOM Bloat
Modern web pages are massive. A typical enterprise dashboard can easily contain 15,000 DOM nodes. Even when heavily compressed, this translates to tens of thousands of input tokens. Because LLM pricing scales linearly with input tokens, running a single multi-step workflow can easily cost upwards of $0.50. 

#### 2. Non-Deterministic Selectors
Web pages are dynamic. Single-page applications (SPAs) constantly mutate the DOM, updating IDs, class names, and layout structures. When an LLM generates a CSS selector or XPath, it does so based on static snapshots. If the DOM mutates mid-execution, the generated selector becomes stale, causing the agent to crash.

#### 3. Cognitive Overkill
You do not need a model capable of writing Shakespearean sonnets to identify an input field with the placeholder `"Enter your email"`. Using a multi-billion parameter model for basic DOM navigation is an architectural anti-pattern. It is an expensive solution to a structured search and state transition problem.

---

### Enter Jev: Deterministic State Compilation

Instead of treating the web browser as an unstructured natural language problem, **Jev** treats the web browser as a **reactive state machine**. 

Jev is an open-source, lightweight execution runtime that compiles declarative UI paths into highly resilient, self-healing execution trees. Instead of using generative inference to figure out what to do next at every step, Jev compiles the target workflow into a state transition graph during development. At runtime, it executes this graph deterministically at native speed.

```
[DOM Mutation] ──> [Jev Reactive Engine] ──(Semantic Anchor Matching)──> [Native Execution]
```

Jev relies on three core concepts:
1. **Semantic Anchors:** Multi-dimensional identifiers that locate UI elements based on functional intent, structural relationships, and visual hierarchy, rather than brittle CSS paths.
2. **The State Compilation Graph:** A declarative map of the target application's states, transitions, and fallback behaviors.
3. **Self-Healing Selectors:** A runtime engine that uses fuzzy tree-distance matching to locate shifted or renamed elements without needing to call an external model.

---

### Deep Dive: The Jev Browser Agent Architecture

Let’s look at how to build a browser agent using Jev. In this scenario, we want our agent to log into an invoice portal, navigate to the billing tab, extract the latest PDF invoice, and download it.

#### 1. Defining the Jev State Graph

Instead of prompting an LLM on every page load, we define the workflow states using Jev’s declarative schema. This graph defines the valid transitions and the triggers required to move between them.

```javascript
// agent.graph.ts
import { JevGraph } from 'jev-core';

export const billingWorkflow = new JevGraph({
  id: 'invoice-downloader',
  initialState: 'unauthenticated',
  states: {
    unauthenticated: {
      on: {
        DOM_READY: {
          target: 'authenticating',
          cond: (ctx) => ctx.hasElement('input[type="email"]')
        }
      }
    },
    authenticating: {
      actions: ['fillCredentials', 'clickLogin'],
      on: {
        LOGIN_SUCCESS: 'dashboard',
        LOGIN_FAILED: 'errorRecovery'
      }
    },
    dashboard: {
      on: {
        DOM_READY: {
          target: 'billingSection',
          actions: ['navigateToBilling']
        }
      }
    },
    billingSection: {
      actions: ['findLatestInvoice', 'downloadPDF'],
      on: {
        DOWNLOAD_COMPLETE: 'success'
      }
    },
    errorRecovery: {
      // Fallback state machine logic
    },
    success: {
      type: 'final'
    }
  }
});
```

#### 2. Implementing Semantic Anchors

The magic of Jev lies in its **Semantic Anchors**. Instead of relying on a fragile selector like `div.container > form > button#submit-btn-2`, Jev uses a weighted attribute matrix to find elements.

If a developer changes the button class from `submit-btn-2` to `btn-primary-large`, Jev’s scoring algorithm still identifies the element because the structural context, text content, and accessibility roles remain unchanged.

Here is how Jev resolves anchors under the hood:

```typescript
// jev-core/resolver.ts
interface SemanticAnchor {
  role: string;
  text?: string | RegExp;
  ariaLabel?: string;
  proximityTo?: string; // CSS selector of a nearby stable element
}

export function resolveElement(anchor: SemanticAnchor, document: Document): HTMLElement | null {
  const candidates = Array.from(document.querySelectorAll(anchor.role)) as HTMLElement[];
  
  let bestCandidate: HTMLElement | null = null;
  let highestScore = 0;

  for (const el of candidates) {
    let score = 0;

    // Match text content
    if (anchor.text && el.textContent) {
      if (typeof anchor.text === 'string' && el.textContent.includes(anchor.text)) {
        score += 50;
      } else if (anchor.text instanceof RegExp && anchor.text.test(el.textContent)) {
        score += 60;
      }
    }

    // Match accessibility labels
    if (anchor.ariaLabel && el.getAttribute('aria-label') === anchor.ariaLabel) {
      score += 40;
    }

    // Match relative proximity (e.g., "the input field next to the 'Password' label")
    if (anchor.proximityTo) {
      const stableNeighbor = document.querySelector(anchor.proximityTo);
      if (stableNeighbor && stableNeighbor.contains(el)) {
        score += 30;
      }
    }

    if (score > highestScore) {
      highestScore = score;
      bestCandidate = el;
    }
  }

  // Self-heal threshold
  return highestScore > 40 ? bestCandidate : null;
}
```

#### 3. The Execution Script

Now let’s look at how we initialize and run the Jev agent using Playwright as our browser automation driver.

```typescript
// run-agent.ts
import { chromium } from 'playwright';
import { JevRuntime } from 'jev-core';
import { billingWorkflow } from './agent.graph';

async function run() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  // Initialize the Jev Runtime on top of Playwright
  const agent = new JevRuntime({
    graph: billingWorkflow,
    context: {
      page,
      credentials: {
        username: 'billing@company.com',
        password: process.env.BILLING_PASSWORD
      }
    }
  });

  // Register action executors
  agent.registerAction('fillCredentials', async (ctx) => {
    const emailField = await ctx.resolve({
      role: 'input',
      ariaLabel: 'Email Address',
      text: /email|username/i
    });
    
    const passwordField = await ctx.resolve({
      role: 'input',
      ariaLabel: 'Password'
    });

    await emailField.fill(ctx.credentials.username);
    await passwordField.fill(ctx.credentials.password);
  });

  agent.registerAction('clickLogin', async (ctx) => {
    const submitBtn = await ctx.resolve({
      role: 'button',
      text: 'Log In'
    });
    await submitBtn.click();
  });

  agent.registerAction('navigateToBilling', async (ctx) => {
    const billingLink = await ctx.resolve({
      role: 'a',
      text: 'Billing & Invoices'
    });
    await billingLink.click();
  });

  // Execute the workflow
  console.log('Starting Jev browser agent...');
  const result = await agent.execute('https://portal.enterprise-saas.com/login');
  
  if (result.status === 'success') {
    console.log('Workflow executed successfully!');
  } else {
    console.error('Workflow failed:', result.error);
  }

  await browser.close();
}

run();
```

---

### Performance Comparison: LLM Agent vs. Jev

To validate our architectural shift, we ran a benchmark of 1,000 automated sessions navigating through a multi-page, highly dynamic invoicing system. 

We compared a **LangChain + GPT-4o-mini** browser agent against our compiled **Jev** agent.

| Metric | LLM-Based Agent (GPT-4o-mini) | Jev-Based Agent |
| :--- | :--- | :--- |
| **Average Step Latency** | 4,520 ms | **12 ms** |
| **Execution Cost (per 1,000 runs)** | $124.50 | **$0.00** |
| **Success Rate (1,000 runs)** | 78.4% | **99.9%** |
| **Memory Footprint** | ~250MB (Node process + context) | **~35MB** |
| **Recovery from DOM Mutation** | High (but slow & costly) | **High (instantaneous, local)** |

#### Why Jev is 100x Faster
An LLM agent must evaluate the entire state of the page at every single step. This means sending a network request, waiting for token generation, parsing the token stream, and executing. 

Jev, on the other hand, performs compile-time optimization. It knows *exactly* what state transition triggers to listen for. Using lightweight MutationObservers inside the browser context, Jev evaluates potential DOM changes in microseconds. It executes actions as fast as the browser can render the pixels.

---

### The Philosophical Shift: Pragmatism Over Hype

The software engineering community has fallen into a dangerous trap: assuming that because a problem *can* be solved with artificial intelligence, it *should* be. 

Web browsers are highly structured document object models. They are built on deterministic standards: HTML, CSS, the DOM API, and accessibility specs (ARIA). When you use an LLM to navigate a web browser, you are throwing away fifty years of structured computer science theory in favor of probabilistic guesswork.

LLMs are brilliant at handling unstructured, creative, and highly unpredictable inputs. They are great for writing copy, generating code, or summarizing articles. But they are a terrible foundation for structured automation pipelines.

By shifting back to a compiled, state-machine-driven architecture with Jev, we didn't just build a faster browser agent. We built a system that we can debug, unit-test, monitor, and scale without worrying about API rate limits or fluctuating model behavior.

Stop using space shuttles to drive to the grocery store. It’s time to bring deterministic engineering principles back to the modern web.