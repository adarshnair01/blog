---
layout: post
title: "Stop Over-Engineering: Why You Must Build Your Own AI Agents From Scratch in Python"
date: 2026-06-22 15:48:49 +0530
excerpt: "Tired of fighting bloated AI frameworks and mysterious black-box behaviors? Learn how to build a production-grade AI agent from scratch using raw Python."
author: "Adarsh Nair"
categories: ai
tags: ["AI", "Tech", "Python", "Software Engineering"]
---

The promise was simple: install a couple of popular open-source frameworks, write five lines of code, and boom—you have an autonomous AI agent capable of managing your schedule, browsing the web, and writing your code. 

But if you have ever tried to take these bloated, over-engineered agentic frameworks into a production environment, you know the reality is far different.

You find yourself trapped in dependency hell. You spend days debugging hidden prompts that you didn’t write. You watch your latency skyrocket as frameworks pass unstructured data through dozens of unnecessary abstractions. When the agent fails, finding the root cause feels like searching for a needle in a digital haystack.

The truth is, you don’t need a heavy, complex framework to build a world-class AI agent. In fact, you shouldn't use one. 

Building an AI agent from scratch in Python is not only surprisingly straightforward, but it also gives you absolute control over execution, latency, and system prompts. This guide will walk you through the "why" and the "how," complete with a production-ready, dependency-free implementation of the ReAct (Reasoning and Acting) architecture.

---

## The Philosophy: What Actually is an AI Agent?

Before writing code, we must strip away the marketing hype. At its core, an AI agent is not a magical sentient entity. It is a **state machine running in a loop**.

An agent consists of four fundamental pillars:
1. **The Brain (LLM):** A language model capable of reasoning, parsing instructions, and formatting outputs.
2. **The Memory (State):** A mechanism to track the history of thoughts, actions, and observations.
3. **The Tools (Functions):** Executable Python code that allows the LLM to interact with the external world (e.g., database queries, web search, APIs).
4. **The Execution Loop (ReAct):** A structured cycle where the model reasons about a problem, decides on an action, executes that action using a tool, observes the result, and repeats until it reaches a final answer.

This pattern, popularized by the seminal paper *ReAct: Synergizing Reasoning and Acting in Language Models*, can be represented by a simple loop:

```
[User Input] ──> ( Thought ──> Action ──> Observation )* ──> [Final Answer]
```

---

## Why Framework-Free is the Ultimate Production Choice

When you build your own agent loop directly in Python, you unlock several critical advantages:

* **Zero Abstraction Overhead:** You write raw Python functions. If a tool fails, you see the exact stack trace. There are no magical wrappers hiding the error.
* **Predictable Token Usage:** Frameworks often inject massive default prompts behind the scenes. Writing your own loop means you control every single token sent to the model.
* **Framework Agnosticism:** You can swap your LLM provider from OpenAI to Anthropic, Cohere, or a local LLaMA instance in minutes without rewriting your entire agent architecture.
* **Lower Latency:** Eliminating intermediate parsing libraries and nested classes significantly speeds up execution times.

---

## Step-by-Step Implementation: Building a ReAct Agent

Let's build a fully functional ReAct agent from scratch. We will use the official `openai` client library to communicate with the model, but we will write the core agent loop ourselves.

### Step 1: Defining the System Prompt

The system prompt is the operating system of your agent. It instructs the model on how to think, how to format its thoughts, and how to call tools. 

We will enforce a strict execution cycle:
* **Thought:** The model's internal reasoning process.
* **Action:** The tool it wants to run, formatted as `tool_name: argument`.
* **Observation:** The result of the tool execution (injected by our Python loop).

```python
SYSTEM_PROMPT = """
You are an advanced AI assistant operating in a loop of Thought, Action, and Observation.
You have access to a set of tools to help you answer the user's query.

Your execution cycle is as follows:
1. Thought: Reason about what you need to do next.
2. Action: Choose a tool to run. Format this exactly as: Action: tool_name: argument
3. Observation: The system will run the tool and return the output.

Available Tools:
- calculate: Solves mathematical equations. Example: calculate: 45 * 12
- get_weather: Retrieves weather info for a city. Example: get_weather: Paris
- database_lookup: Searches employee database. Example: database_lookup: John Doe

Rules:
- You must write "Thought:" before reasoning.
- You must write "Action:" followed by "tool_name: argument" when you need to use a tool.
- Do not invent observations. Wait for the loop to provide them.
- When you have the final answer, respond directly to the user without writing "Action:".

Begin!
"""
```

### Step 2: Coding the Tools

Tools are just standard Python functions. Let's create a dictionary mapping tool names to actual executable functions.

```python
import json

def calculate(expression: str) -> str:
    try:
        # Safe evaluation of basic math operations
        allowed_chars = "0123456789+-*/(). "
        if not all(char in allowed_chars for char in expression):
            return "Error: Invalid characters in expression."
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def get_weather(city: str) -> str:
    # A mock weather database
    weather_data = {
        "london": "Rainy, 14°C",
        "paris": "Sunny, 21°C",
        "new york": "Windy, 18°C"
    }
    return weather_data.get(city.lower().strip(), "Weather data not found for this location.")

def database_lookup(name: str) -> str:
    # A mock employee database
    db = {
        "john doe": "Role: Senior Engineer, Team: Platform, Tenure: 4 years",
        "jane smith": "Role: Product Manager, Team: Growth, Tenure: 2 years"
    }
    return db.get(name.lower().strip(), "No record found in database.")

# Map tool names to their respective functions
TOOLS = {
    "calculate": calculate,
    "get_weather": get_weather,
    "database_lookup": database_lookup
}
```

### Step 3: Parsing the Model's Intentions

To make the loop work, we need a robust parser that can extract the `Action` and `argument` from the model’s raw text response. We will use regular expressions for this.

```python
import re

def parse_action(model_output: str):
    # Regex to find the pattern Action: tool_name: argument
    match = re.search(r"Action:\s*(\w+):\s*(.*)", model_output)
    if match:
        tool_name = match.group(1).strip()
        argument = match.group(2).strip()
        return tool_name, argument
    return None, None
```

### Step 4: The Core Agent Execution Loop

Now, we tie everything together inside a simple `while` loop. The loop sends the conversation history to the LLM, reads the response, checks if a tool call was requested, executes the tool, appends the result as an `Observation`, and continues until the model decides to stop calling tools.

```python
import os
from openai import OpenAI

# Initialize the OpenAI client (ensure OPENAI_API_KEY is set in your environment variables)
client = OpenAI()

def run_agent(user_query: str, max_iterations: int = 5):
    # Initialize the conversation state with the system prompt and the user query
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    print(f"Starting Agent for query: '{user_query}'\n")
    
    for iteration in range(max_iterations):
        print(f"--- Iteration {iteration + 1} ---")
        
        # Call the LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0 # Keep temperature at 0 for deterministic tool usage
        )
        
        assistant_response = response.choices[0].message.content
        print(assistant_response)
        
        # Append the assistant's thoughts/actions to history
        messages.append({"role": "assistant", "content": assistant_response})
        
        # Check if the model requested an action
        tool_name, argument = parse_action(assistant_response)
        
        if tool_name:
            if tool_name in TOOLS:
                print(f"\n[Executing Tool: {tool_name} with arg: '{argument}']")
                # Execute the tool
                observation = TOOLS[tool_name](argument)
                print(f"[Observation: {observation}]\n")
                
                # Feed the observation back to the assistant
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                error_msg = f"Error: Tool '{tool_name}' does not exist."
                print(f"[{error_msg}]")
                messages.append({"role": "user", "content": f"Observation: {error_msg}"})
        else:
            # If no action was parsed, the model has provided its final answer
            print("\nAgent finished execution.")
            return assistant_response
            
    print("\nAgent stopped: Reached maximum iterations.")
    return "I was unable to resolve your request within the iteration limit."
```

### Step 5: Testing Our Agent

Let's run a query that requires the agent to combine multiple tools to find the answer. We will ask it to look up an employee's tenure, double it, and then check the weather in Paris.

```python
if __name__ == "__main__":
    query = "Find John Doe's tenure, multiply it by 3, and check the weather in Paris."
    run_agent(query)
```

### Expected Output Trace:

```text
Starting Agent for query: 'Find John Doe's tenure, multiply it by 3, and check the weather in Paris.'

--- Iteration 1 ---
Thought: I need to find John Doe's tenure first. I will use the database_lookup tool for this.
Action: database_lookup: John Doe

[Executing Tool: database_lookup with arg: 'John Doe']
[Observation: Role: Senior Engineer, Team: Platform, Tenure: 4 years]

--- Iteration 2 ---
Thought: John Doe's tenure is 4 years. Now I need to multiply this by 3. I will use the calculate tool.
Action: calculate: 4 * 3

[Executing Tool: calculate with arg: '4 * 3']
[Observation: 12]

--- Iteration 3 ---
Thought: The result of the calculation is 12. Next, I need to check the weather in Paris. I will use the get_weather tool.
Action: get_weather: Paris

[Executing Tool: get_weather with arg: 'Paris']
[Observation: Sunny, 21°C]

--- Iteration 4 ---
Thought: I have all the required information. John Doe's tenure is 4 years, which multiplied by 3 is 12. The weather in Paris is Sunny, 21°C. I can now compile the final answer.
John Doe has a tenure of 4 years. Multiplying this by 3 gives 12. Meanwhile, the weather in Paris is currently Sunny and 21°C.

Agent finished execution.
```

---

## Moving to Production: Crucial Guardrails

While this lightweight agent loop is highly efficient, taking it to production requires a few structural guardrails:

1. **Infinite Loop Protection:** Always enforce a `max_iterations` limit (as we did above) to prevent the agent from getting stuck in a cycle of bad tool calls.
2. **Strict Schema Parsing:** For complex production environments, consider using Pydantic models to validate arguments passed to your tools.
3. **Async Execution:** If your tools involve external API calls, convert your tools and execution loop to use Python's `asyncio` to handle concurrent tasks efficiently.
4. **Token Budgeting:** Monitor the length of the `messages` history. If the agent runs for many iterations, implement a sliding window or summarization strategy to prune older messages.

## Conclusion

By writing your own AI agent from scratch, you pull back the curtain on one of the most talked-about paradigms in modern technology. You realize that agents aren't magic—they are simply clean engineering, structured prompting, and reliable loop execution. 

The next time you build an agentic workflow, skip the framework install. Write it yourself. You will gain speed, stability, and a deep architectural understanding that no pre-packaged library can ever provide.