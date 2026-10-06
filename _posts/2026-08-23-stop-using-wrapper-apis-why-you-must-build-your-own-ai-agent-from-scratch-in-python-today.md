---
layout: post
title: "Stop Using Wrapper APIs: Why You Must Build Your Own AI Agent from Scratch in Python Today"
date: 2026-08-23 20:26:48 +0530
excerpt: "Tired of generic wrapper apps and bloated orchestration frameworks? Learn how to code a resilient, autonomous AI agent from scratch using pure Python."
author: "Adarsh Nair"
categories: ai
tags: ["AI", "Python", "Agents", "Machine Learning", "Tutorial"]
---

# Stop Using Wrapper APIs: Why You Must Build Your Own AI Agent from Scratch in Python Today

Let’s be honest: half of the "AI engineering" happening right now is just wrapping someone else's API in a web UI and calling it a day. 

We import massive, opinionated frameworks that abstract away the exact mechanics we need to understand. We build systems that hallucinate loops, burn through hundreds of dollars in API credits on infinite thought-chains, and break the moment an underlying model updates its JSON schema format.

If you want to truly master the paradigm shift of the generative AI era, you need to stop hiding behind wrappers. You need to build your AI agent **completely from scratch in Python**. 

In this comprehensive guide, we are stripping away LangChain, LlamaIndex, and every other heavy abstraction layer. We are going to build a production-ready, autonomous ReAct (Reasoning and Acting) agent using raw Python, standard libraries, and direct LLM API calls. 

By the end of this post, you won't just understand how agents work—you'll command them.

---

## The Anatomy of an AI Agent: What Are We Actually Building?

Before writing a single line of code, let's deconstruct what an "agent" actually is. 

At its core, an AI agent is simply a feedback loop composed of four distinct pillars:
1. **The Brain (LLM):** The reasoning engine that processes state and decides what to do next.
2. **The Prompt/Context:** Instructions, system prompts, and memory that guide the model's behavior.
3. **Tools:** Functions or APIs the model can execute to interact with the real world (e.g., calculators, web scrapers, database queries).
4. **The Execution Loop:** A programmatic structure that takes the LLM's output, checks if it wants to use a tool, executes that tool, feeds the result back into the context, and repeats until a final answer is reached.

The most popular architecture for this is the **ReAct (Reasoning + Acting)** framework. The model alternates between a `Thought` (internal monologue), an `Action` (calling a tool), and an `Observation` (the output of that tool).

```
[User Input] --> [LLM Brain: Thought & Action] --> [Tool Execution]
                        ^                                   |
                        |------- [Observation Feedback] <---|
```

Let's write the code to bring this loop to life.

---

## Step 1: Setting Up the Environment

We’ll keep dependencies to an absolute minimum. You only need Python 3.10+ and the official client library for your preferred LLM provider (we'll use `openai` for this example, but any provider with function calling works).

```bash
pip install openai python-dotenv
```

Create a `.env` file in your project root:
```env
OPENAI_API_KEY=your_api_key_here
```

---

## Step 2: Defining the Tools

An agent without tools is just a chatbot with extra steps. Let's create two practical tools: a live web search simulation (or calculator) and a file-system writer. For simplicity and reliability in this tutorial, we will write a Python function that performs mathematical calculations and another that fetches current weather data via a public API.

```python
import json
import requests
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define our tools as standard Python functions
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression safely."""
    try:
        # Using eval with extreme caution for demo purposes
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating math: {str(e)}"

def get_weather(city: str) -> str:
    """Fetches the current weather for a given city."""
    # Mocking an API call for resilience
    weather_data = {
        "london": "15°C, Partly Cloudy",
        "new york": "22°C, Sunny",
        "tokyo": "18°C, Rainy"
    }
    city_lower = city.strip().lower()
    return weather_data.get(city_lower, f"Weather data not found for {city}.")

# Map tool names to actual Python functions
AVAILABLE_TOOLS = {
    "calculate": calculate,
    "get_weather": get_weather
}
```

---

## Step 3: Configuring OpenAI Function Calling Schemas

LLMs don't magically know how to run Python code unless we explicitly tell them what tools are available and what parameters those tools expect. We define this using JSON schemas.

```python
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "get_weather": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g., '25 * 4'."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city, e.g., 'Tokyo'."
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

---

## Step 4: Building the Autonomous Execution Loop

This is the beating heart of our agent. The loop sends the conversation history to the model, inspects the response to see if the model requested a tool call, executes that tool locally, appends the result to the history, and repeats until the model provides a final text response.

```python
def run_agent(user_prompt: str, max_iterations: int = 5):
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant with access to tools. Use them when necessary."
        },
        {
            "role": "user", 
            "content": user_prompt
        }
    ]

    print(f"\n[User Query]: {user_prompt}\n" + "-"*40)

    for iteration in range(max_iterations):
        print(f"--- Iteration {iteration + 1} ---")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)

        # Check if the model wants to call a tool
        if response_message.tool_calls:
            print("Model decided to use a tool.")
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f" > Calling tool: `{function_name}` with args: {function_args}")
                
                # Execute the corresponding local python function
                if function_name in AVAILABLE_TOOLS:
                    tool_function = AVAILABLE_TOOLS[function_name]
                    tool_output = tool_function(**function_args)
                else:
                    tool_output = f"Error: Tool {function_name} not found."
                
                print(f" < Tool Output: {tool_output}")
                
                # Feed the tool output back into the conversation history
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_output
                })
        else:
            # If no tool call was requested, the model has its final answer
            print("\n[Final Answer Reached]:")
            return response_message.content

    return "Agent stopped: Maximum iterations reached without final answer."

# Let's test our agent!
if __name__ == "__main__":
    query = "What is the weather in Tokyo right now, and if I multiply that temperature value by 2, what do I get?"
    final_output = run_agent(query)
    print(final_output)
```

---

## Step 5: Why This Approach Beats Heavy Frameworks

When you run this script, notice what just happened:
1. **Total Transparency:** You can see every single token, system prompt adjustment, tool invocation, and error handling step. There is no hidden black box.
2. **Infinite Customizability:** Need to inject custom logging, persistent memory using SQLite, or human-in-the-loop approval gates? Just write standard Python code around the loop. 
3. **Zero Bloat:** Your application starts instantly, has minimal dependencies, and won't break because a third-party framework updated its API definitions overnight.

---

## Conclusion: Take Back Control of Your AI Stack

Frameworks like LangChain have their place for quick prototyping, but true engineering mastery comes from understanding the primitive building blocks beneath the abstractions. By building your AI agent from scratch in Python, you unlock the ability to debug complex systemic behaviors, optimize token usage down to the byte, and build resilient applications that actually scale in production.

Now close the browser tabs, fire up your IDE, and start hacking away at your own custom loop.