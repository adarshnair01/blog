---
layout: post
title: "Stop Using LangChain: Why Building an AI Agent from Scratch in Python is Your New Superpower"
date: 2026-06-20 08:07:03 +0530
excerpt: "Think you need bloated frameworks to build autonomous AI? Think again. Here is how to build a production-grade ReAct agent from scratch in under 100 lines of pure Python."
author: "Adarsh Nair"
categories: ai
tags: ["AI", "Tech", "Python", "LLM", "Software Engineering"]
---

We are currently living through a massive gold rush in generative AI. Every week, a new framework emerges promising to make AI agent development as simple as writing three lines of configuration. We’ve seen the rise of massive orchestrators like LangChain, CrewAI, and AutoGen. 

But behind the shiny marketing and GitHub star counts lies a frustrating reality: **framework fatigue**. 

Many developers who start building production-grade agents with these libraries quickly run into a brick wall. The abstractions are too thick. Debugging a simple loop requires tracing through ten layers of nested library code. Customizing agent behavior feels like fighting the framework rather than writing software. 

The truth is, you don’t need a massive dependency tree to build a state-of-the-art AI agent. In fact, building an agent from scratch in pure Python is not only surprisingly simple—it is also the best way to gain complete control over your AI's latency, cost, and execution path.

In this guide, we will break down the philosophy of the ReAct (Reasoning and Acting) paradigm, design a robust agent architecture, and implement a fully functional, production-ready AI agent in under 100 lines of Python.

---

## The Core Philosophy: What Actually is an AI Agent?

Before writing code, we must strip away the buzzwords. At its absolute core, an AI agent is simply a design pattern. 

Unlike a standard LLM chat completion (which is a single, static input-to-output mapping), an agent operates in a **closed-loop system**. It has access to tools, can make decisions based on execution outcomes, and runs in a continuous cycle until it determines its goal has been met.

This is best summarized by the **ReAct (Reason + Act)** framework, pioneered by Yao et al. in 2022. The cycle works like this:

1. **Thought:** The LLM analyzes the user's query and decides what it needs to do.
2. **Action:** The LLM chooses a specific tool and provides the necessary arguments.
3. **Observation:** The system executes the tool, captures the output, and feeds it back to the LLM.
4. **Repeat:** The LLM reviews the new information, formulates a new thought, and either takes another action or returns the final answer.

```
+--------------------------------------------------+
|                     User Query                   |
+------------------------+-------------------------+
                         |
                         v
+------------------------+-------------------------+
|                      Agent                       |
|  +--------------------------------------------+  |
|  |                  Thought                   |  |
|  +---------------------+----------------------+  |
|                        |                         |
|                        v                         |
|  +---------------------+----------------------+  |
|  |                  Action                    |  |
|  +---------------------+----------------------+  |
+------------------------|-------------------------+
                         | (Executes Tool)
                         v
+------------------------+-------------------------+
|                   Tool Execution                 |
|  (e.g., Database Query, Web Search, Calculator)  |
+------------------------+-------------------------+
                         |
                         v (Observation)
+------------------------+-------------------------+
|                      Agent                       |
|  +---------------------+----------------------+  |
|  |                Observation                 |  |
|  +---------------------+----------------------+  |
|                        |                         |
|                        v                         |
|                  [Repeat Loop]                   |
+------------------------+-------------------------+
                         | (Goal Reached)
                         v
+------------------------+-------------------------+
|                    Final Answer                  |
+--------------------------------------------------+
```

By implementing this loop ourselves, we eliminate magic and gain absolute transparency over our agent's state machine.

---

## Step 1: Defining the Tools

Our agent needs to interact with the outside world. Let’s define two simple, deterministic Python functions that our agent can call: a basic calculator and a mock weather API.

```python
import json
import math

def calculate(expression: str) -> str:
    """Safely evaluates a basic mathematical expression."""
    try:
        # Use a safe evaluation environment
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        return str(eval(expression, {"__builtins__": None}, allowed_names))
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def get_weather(location: str) -> str:
    """Returns the mock weather for a given city."""
    loc = location.lower()
    if "tokyo" in loc:
        return "Tokyo is currently 18°C and rainy."
    elif "san francisco" in loc:
        return "San Francisco is currently 14°C and foggy."
    elif "paris" in loc:
        return "Paris is currently 22°C and sunny."
    else:
        return f"Could not find weather data for '{location}'."

# A registry mapping tool names to actual functions
TOOL_REGISTRY = {
    "calculate": calculate,
    "get_weather": get_weather
}
```

---

## Step 2: Crafting the System Prompt

The system prompt is the engine of our agent. It instructs the LLM on how to think, format its output, and interact with the tools. We use a strict parsing structure so our Python code can reliably extract the agent's intent.

```python
SYSTEM_PROMPT = """
You are an advanced, autonomous AI Agent. You solve problems by thinking step-by-step and utilizing tools.

You have access to the following tools:
- calculate: Evaluates mathematical expressions. Input must be a valid Python math string. Example: "calculate: 2 + 2" or "calculate: math.sqrt(16)"
- get_weather: Retrieves weather data for a city. Input must be a city name. Example: "get_weather: Tokyo"

Your execution flow MUST follow this exact structure:

Thought: [Your reasoning about what to do next]
Action: [tool_name: tool_input]
Observation: [The system will provide this result]

... (this Thought/Action/Observation cycle can repeat multiple times)

Thought: I have gathered all necessary information.
Final Answer: [The ultimate, comprehensive response to the user]

Let's begin!
"""
```

---

## Step 3: Designing the Execution Loop

Now, we write the loop. We will use the official `openai` SDK (v1.0.0+). The loop will send the conversation history to the LLM, parse the output for an `Action`, execute the corresponding tool, append the `Observation` to the message history, and repeat.

```python
import re
from openai import OpenAI

client = OpenAI()

def parse_action(text: str):
    """Searches for Action: tool_name: tool_input in the text."""
    match = re.search(r"Action:\s*(\w+):\s*(.*)", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None

def run_agent(user_prompt: str, max_turns: int = 5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    print(f"[*] Starting agent with query: '{user_prompt}'\n")
    
    for turn in range(1, max_turns + 1):
        print(f"--- TURN {turn} ---")
        
        # Step 1: Query the LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0 # Force deterministic output
        )
        
        assistant_text = response.choices[0].message.content
        print(assistant_text)
        
        # Keep track of the conversation
        messages.append({"role": "assistant", "content": assistant_text})
        
        # Step 2: Check if the agent wants to take an action
        tool_name, tool_input = parse_action(assistant_text)
        
        if tool_name and tool_input:
            if tool_name in TOOL_REGISTRY:
                print(f"\n[System] Executing Tool '{tool_name}' with input: '{tool_input}'")
                observation = TOOL_REGISTRY[tool_name](tool_input)
                print(f"[System] Observation: {observation}\n")
                
                # Feed the observation back into the loop
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                error_msg = f"Tool '{tool_name}' is not available."
                print(f"\n[System] Error: {error_msg}\n")
                messages.append({"role": "user", "content": f"Observation: {error_msg}"})
        
        # Step 3: Check if the final answer has been provided
        if "Final Answer:" in assistant_text:
            print("\n[*] Agent successfully finished execution.")
            break
    else:
        print("\n[!] Agent terminated: Maximum turns reached without a final answer.")
```

---

## Step 4: Running the Agent

Let's test our agent with a complex query that requires sequential reasoning and tool usage: fetching the weather in Tokyo and calculating a mathematically modified value based on it.

```python
if __name__ == "__main__":
    query = "What is the weather in Tokyo? If we triple that temperature, what would it be?"
    run_agent(query)
```

### The Output Trace:

```text
[*] Starting agent with query: 'What is the weather in Tokyo? If we triple that temperature, what would it be?'

--- TURN 1 ---
Thought: I need to find the current temperature in Tokyo first. I will use the `get_weather` tool for this.
Action: get_weather: Tokyo

[System] Executing Tool 'get_weather' with input: 'Tokyo'
[System] Observation: Tokyo is currently 18°C and rainy.

--- TURN 2 ---
Thought: The weather in Tokyo is 18°C. Now I need to triple this temperature, which means calculating 18 * 3. I will use the `calculate` tool.
Action: calculate: 18 * 3

[System] Executing Tool 'calculate' with input: '18 * 3'
[System] Observation: 54

--- TURN 3 ---
Thought: I have gathered all necessary information.
Final Answer: The current temperature in Tokyo is 18°C. If you triple that temperature, it would be 54°C.

[*] Agent successfully finished execution.
```

---

## Why This Simple Design Wins in Production

By stripping away the abstraction layers, we unlock massive advantages for production deployments:

1. **Deterministic State Management:** Since the agent history is a simple list of dictionaries, saving and resuming agent sessions is trivial. You can save the session state directly to Redis or PostgreSQL as a JSON blob.
2. **Infinite Customization:** Want to add semantic routing? Want to inject dynamic tools based on user permissions? You can write plain Python code to do this. No need to subclass complex framework objects.
3. **Optimized Latency:** Frameworks often introduce hidden API calls or prompt bloating. Writing your own loop ensures you are only sending the tokens you explicitly want to send.
4. **Frictionless Debugging:** If your agent gets stuck in a loop, you can simply set a breakpoint (`import pdb; pdb.set_trace()`) directly in your loop. You don't have to navigate through deep call stacks inside a third-party package.

## Conclusion

Frameworks are great for rapid prototyping, but they can obscure the underlying mechanics of AI engineering. 

By building your agent loop from scratch, you gain a deep, intuitive understanding of how LLMs interact with code. You realize that agents aren't magic—they are simply state machines wrapped around an LLM.

The next time you start a new AI project, resist the urge to immediately install a bloated ecosystem. Start with a simple loop, write pure Python, and build exactly what you need.