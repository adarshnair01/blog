---
layout: post
title: "Stop Using Wrapper APIs: Why Building an AI Agent From Scratch in Python Will Completely Change How You Code"
date: 2026-09-08 09:38:41 +0530
excerpt: "Tired of relying on fragile third-party frameworks? Here is the raw, unvarnished blueprint to building an autonomous AI agent from absolute scratch in Python."
author: "Adarsh Nair"
categories: ai
tags: ["AI", "Python", "Machine Learning", "Software Architecture"]
---

We have all been there. You pip install the latest trendy wrapper framework, wrap your head around its hyper-opinionated abstraction layers, and three days later, you hit a brick wall. The framework’s built-in state manager breaks, the custom tool-calling syntax suddenly deprecates, and you are left debugging code you didn't write to solve a problem you barely understand.

It is time to strip away the magic. 

If you want to truly understand how artificial intelligence moves from a passive text-generator into an active digital worker, you need to build an AI agent from absolute scratch in Python. No heavy frameworks. No black boxes. Just pure logic, raw API calls, and a loop of reasoning that mimics cognition.

By the end of this deep-dive, you won't just have built a functional autonomous agent—you will possess a fundamental mastery of agentic architecture that puts you in the top 1% of AI developers.

---

## What Actually *Is* an AI Agent? (Breaking the Illusion)

Let’s demystify the buzzword. People throw around "AI Agent" like it’s a sentient piece of software. In reality, an agent is just a Large Language Model (LLM) trapped inside a `while` loop with access to tools and a memory stack.

```
+-------------------------------------------------------+
|                    The Agent Loop                     |
|                                                       |
|   +----------+      +------------+      +---------+   |
|   |   User   | ---> | Perception | ---> | Memory  |   |
|   +----------+      +------------+      +---------+   |
|                          |                   |        |
|                          v                   v        |
|                    +-----------------------------+    |
|                    |     LLM "Brain" (Reason)    |    |
|                    +-----------------------------+    |
|                          |                            |
|                          v                            |
|                     [Action?]                         |
|                    /         \                        |
|                 (Yes)        (No)                     |
|                  /             \                      |
|        +-------------+       +-----------+            |
|        | Execute Tool|       | Final     |            |
|        +-------------+       | Response  |            |
|               |              +-----------+            |
|               v                                       |
|         (Feed back to Loop)                           |
+-------------------------------------------------------+
```

An LLM on its own is stateless. It receives an input, predicts the next best tokens, and stops. It cannot check the live weather, it cannot query a SQL database, and it cannot execute a bash script unless you give it the scaffolding to do so. 

An **Agent** introduces three core pillars:
1. **Perception/Memory:** The ability to retain context and remember past actions.
2. **Reasoning:** Using the LLM as a cognitive engine to determine *what* needs to be done next.
3. **Action:** The ability to invoke external functions (tools) based on its reasoning, observe the output, and iterate until the goal is achieved.

Let's build this from the ground up using pure Python.

---

## Step 1: Setting Up the Environment

We don't need LangChain, LlamaIndex, or any bloated dependencies. We just need a way to talk to an LLM provider (we'll use OpenAI's native client, but this works identically with Anthropic or local models via Ollama) and basic Python typing.

Fire up your terminal:

```bash
pip install openai pydantic python-dotenv
```

Create a `.env` file and drop your API key in:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## Step 2: Defining the Tool Architecture

An agent is only as powerful as the tools it can wield. In modern agentic systems, we define tools using standard Python functions, leveraging type hints and docstrings so the LLM understands *what* the tool does, *what* parameters it expects, and *why* it should use it.

Let's write two simple tools: a web search mock (or basic calculator) and a file writer. For demonstration clarity, let's build a secure calculator and a system status checker.

```python
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluate_expression(expression: str) -> str:
    """
    Evaluates a mathematical expression safely.
    Args:
        expression: A string mathematical expression like '45 * 23'
    """
    try:
        # Using eval safely in a controlled context for demo purposes
        allowed_characters = set("0123456789+-*/.() ")
        if not all(char in allowed_characters for char in expression):
            return "Error: Invalid characters in expression."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def get_system_load() -> str:
    """Returns simulated system performance metrics."""
    return "CPU Load: 14.2%, Memory Usage: 68.4%, Status: Optimal"

# Map function names to actual callables
available_tools = {
    "evaluate_expression": evaluate_expression,
    "get_system_load": get_system_load
}
```

---

## Step 3: Crafting OpenAI Tool Schemas

To let the LLM know these tools exist, we must serialize their definitions into JSON schemas that OpenAI expects. While frameworks do this automatically using introspection, doing it explicitly gives us total structural control.

```python
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_expression",
            "description": "Evaluates a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '1024 * 4'."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_load",
            "description": "Retrieves current system health and performance metrics.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]
```

---

## Step 4: Building the Reasoning and Execution Loop (The Core Engine)

This is where the magic happens. The agent needs a conversational history (memory) so it remembers what it has tried. It sends messages to the model, checks if the model wants to call a tool, executes that tool locally, appends the result back to the message history, and loops until the model gives a final text answer.

```python
class ScratchAgent:
    def __init__(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]

    def interact(self, user_prompt: str):
        self.messages.append({"role": "user", "content": user_prompt})
        
        iteration = 0
        max_iterations = 5

        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Agent Iteration {iteration} ---")

            # Call the LLM
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=self.messages,
                tools=tools_schema,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            
            # Append model's response to conversation history
            self.messages.append(response_message)

            # Check if the model wants to call a tool
            if response_message.tool_calls:
                print(f"Model decided to use tools: {len(response_message.tool_calls)} call(s)")
                
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"-> Executing tool: {function_name} with args: {function_args}")
                    
                    if function_name in available_tools:
                        tool_function = available_tools[function_name]
                        # Execute local python function
                        tool_output = tool_function(**function_args)
                    else:
                        tool_output = f"Error: Tool {function_name} not found."

                    print(f"<- Tool output: {tool_output}")

                    # Feed the tool execution result back into the agent's memory
                    self.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(tool_output),
                    })
            else:
                # No tool calls requested; the model has its final answer
                return response_message.content

        return "Agent stopped: Reached maximum iteration limit without final answer."
```

---

## Step 5: Testing Your Scratch-Built Agent

Let's initialize our agent and give it a multi-step problem that requires checking system specs and running calculations.

```python
if __name__ == "__main__":
    prompt = (
        "Check our system health first. Then, calculate what 8493 multiplied by "
        "382, and finally, summarize the system status alongside the calculation result."
    )
    
    agent = ScratchAgent(
        system_prompt="You are a helpful, precise DevOps assistant equipped with diagnostic and math tools."
    )
    
    final_answer = agent.interact(prompt)
    
    print("\n====================")
    print("FINAL AGENT RESPONSE:")
    print("====================")
    print(final_answer)
```

### Run the Code and Watch It Think
When you run this script, you will see your terminal come alive:
1. The model analyzes your prompt.
2. It decides it needs system metrics and calls `get_system_load`.
3. It receives the output, realizes it also needs a multiplication calculation, and calls `evaluate_expression`.
4. It compiles both results into a coherent, comprehensive final answer.

---

## Why This Matters (The Ultimate Takeaway)

By building an agent this way, you bypass the learning curve of heavy abstractions. You understand precisely how memory arrays are structured, how tool-calling IDs map back to LLM responses, and how context windows inflate over multiple iterations.

When a framework breaks tomorrow, you won't panic—because you didn't build your career on someone else's wrapper. You built your understanding from the metal up.