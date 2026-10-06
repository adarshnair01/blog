---
layout: post
title: "Your AI Agents Are Stealing Your Users' Data (And How Amazon Bedrock AgentCore Fixes This Security Nightmare)"
date: 2026-06-25 09:27:33 +0530
excerpt: "Autonomous AI agents are a security disaster waiting to happen. Learn how to secure your enterprise agents using Amazon Bedrock AgentCore and OAuth consent management."
author: "Adarsh Nair"
categories: ai
tags: ["AI", "Tech", "AWS", "Security"]
---

We are living in the wild west of agentic AI. 

Every developer and enterprise is rushing to build autonomous AI agents that can browse the web, write code, access databases, and interact with third-party APIs. We want our agents to schedule meetings on Google Calendar, update pipelines in Salesforce, and trigger deployments on GitHub. 

But there is a massive, gaping security hole in this vision: **How do you securely manage end-user authorization?**

If your AI agent has direct, unconstrained access to your user's third-party accounts, you aren't building a helper—you’re building a security nightmare. If an LLM is hijacked via prompt injection, it could easily abuse those credentials to exfiltrate sensitive data or perform unauthorized actions on behalf of the user.

To build production-grade, secure agents, you must implement robust, fine-grained end-user OAuth consent. 

In this deep dive, we will explore how to manage end-user OAuth consent for AI agents using **Amazon Bedrock AgentCore**—the core runtime and orchestration framework driving Amazon Bedrock Agents. We’ll look at the architectural patterns, security handshakes, and concrete code implementations required to keep your users' data safe.

---

## The Core Challenge: Why Traditional OAuth Fails in Agentic Workflows

In a traditional web application, OAuth is straightforward:
1. The user clicks "Connect to Slack."
2. The user is redirected to Slack's authorization server.
3. The user grants permissions and is redirected back with an authorization code.
4. The backend exchanges the code for an access token and stores it in a secure session or database.
5. Future API requests use this token.

But AI agents introduce dynamic, non-deterministic execution paths. An agent powered by a Large Language Model (LLM) decides *on the fly* which tools to call based on the user's prompt. 

If a user says, *"Check my calendar for tomorrow and draft an email to Alice,"* the agent must:
1. Realize it needs access to Google Calendar.
2. Determine if it has a valid OAuth token for this specific user.
3. If it doesn't, pause execution, safely prompt the user to authenticate, and resume execution once authorized.
4. Execute the action securely *without* exposing the underlying access token to the LLM's context window (where it could be leaked).

This is where **Amazon Bedrock AgentCore** comes in. It provides the runtime orchestration engine to intercept tool executions, manage state, and handle external authorization flows seamlessly.

---

## The Architecture: Secure OAuth Handshake with Amazon Bedrock

To manage OAuth consent without exposing tokens to the LLM, we use an architecture that decouples the **agent's reasoning engine** from the **execution environment**.

Here is how the secure OAuth consent flow works using Amazon Bedrock Agents, AWS Lambda, and AWS Secrets Manager/DynamoDB:

```
[ User ] <---> [ Client App ] <=========================> [ Amazon Bedrock Agent ]
   |                ^                                             |
   |                | 2. Intercepts action & requests auth        |
   |                v                                             v
   |         [ Orchestrator / API Gateway ] <===========> [ Action Group Lambda ]
   |                |                                             |
   | 3. Redirect    | 4. Stores securely                          | 1. Checks for 
   v                v                                             |    valid token
[ Identity Provider (IdP) ]                                       v
                                                           [ Secrets Manager ]
```

### The Step-by-Step Flow:
1. **User Request**: The user asks the agent to perform an action requiring third-party access (e.g., *"Sync my Jira board"*).
2. **Action Check**: The Bedrock Agent invokes the Action Group Lambda function.
3. **Token Verification**: The Lambda function checks if a valid OAuth access/refresh token exists for this specific user ID in AWS Secrets Manager or an encrypted DynamoDB table.
4. **Consent Required (The Breakout)**: 
   * If no token exists, the Lambda function returns a structured response using Bedrock's `returnControl` payload.
   * The Agent pauses its current execution state and sends an authentication URI back to the client application.
5. **User Consent**: The client application redirects the user to the third-party Identity Provider (IdP) to authenticate and grant consent.
6. **Token Capture**: The callback handler receives the authorization code, exchanges it for tokens, encrypts them, and saves them securely.
7. **Resume Execution**: The client app notifies the Bedrock Agent to resume the session. The agent runs the Action Group Lambda again, which now successfully retrieves the token and executes the API call.

---

## Step-by-Step Implementation

Let's write some code. We will build an Action Group for an Amazon Bedrock Agent that interacts with a third-party service (e.g., GitHub) using OAuth 2.0.

### 1. Defining the OpenAPI Schema for the Action Group

Amazon Bedrock Agents rely on OpenAPI schemas to understand what tools are available. We need to define an action that requires user authorization.

```json
{
  "openapi": "3.0.1",
  "info": {
    "title": "GitHub Integration Agent",
    "version": "1.0.0",
    "description": "Allows the agent to interact with GitHub on behalf of the user."
  },
  "paths": {
    "/create-issue": {
      "post": {
        "summary": "Create a GitHub Issue",
        "description": "Creates a new issue in a specified GitHub repository.",
        "operationId": "createGitHubIssue",
        "parameters": [
          {
            "name": "userId",
            "in": "header",
            "required": true,
            "schema": {
              "type": "string"
            },
            "description": "The unique identifier of the end user."
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "repo": { "type": "string", "description": "Format: owner/repo" },
                  "title": { "type": "string" },
                  "body": { "type": "string" }
                },
                "required": ["repo", "title"]
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Issue created successfully"
          },
          "401": {
            "description": "Unauthorized. OAuth consent required."
          }
        }
      }
    }
  }
}
```

### 2. Writing the Action Group Lambda Handler

This AWS Lambda function intercepts the agent's request. If the user hasn't authorized the application yet, it explicitly tells the Bedrock Agent runtime that OAuth consent is required.

```python
import json
import os
import boto3
from botocore.exceptions import ClientError
import requests

secrets_client = boto3.client('secretsmanager')

def get_user_token(user_id):
    """Retrieves the OAuth access token from Secrets Manager."""
    try:
        secret_name = f"user/oauth/{user_id}"
        response = secrets_client.get_secret_value(SecretId=secret_name)
        secret_data = json.loads(response['SecretString'])
        return secret_data.get('access_token')
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            return None
        raise e

def lambda_handler(event, context):
    agent = event['agent']
    action_group = event['actionGroup']
    api_path = event['apiPath']
    
    # Extract userId from request headers/parameters
    user_id = None
    for param in event.get('parameters', []):
        if param['name'] == 'userId':
            user_id = param['value']
            
    if not user_id:
        return create_response(event, 400, {"message": "Missing userId parameter."})

    # 1. Check for valid OAuth Token
    access_token = get_user_token(user_id)
    
    if not access_token:
        # 2. No token found: Initiate OAuth Consent Breakout
        auth_url = f"https://github.com/login/oauth/authorize?client_id={os.environ['GITHUB_CLIENT_ID']}&state={user_id}&scope=repo"
        
        # We return a 401 and provide the authorization URI to the client application
        return create_response(
            event, 
            401, 
            {
                "message": "Authorization required.",
                "auth_url": auth_url
            }
        )

    # 3. Token exists: Execute the secure API call on behalf of the user
    if api_path == '/create-issue':
        body = event['requestBody']['content']['application/json']['properties']
        repo = next(p['value'] for p in body if p['name'] == 'repo')
        title = next(p['value'] for p in body if p['name'] == 'title')
        issue_body = next((p['value'] for p in body if p['name'] == 'body'), "")
        
        headers = {
            "Authorization": f"token {access_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        response = requests.post(
            f"https://api.github.com/repos/{repo}/issues",
            json={"title": title, "body": issue_body},
            headers=headers
        )
        
        if response.status_code == 201:
            return create_response(event, 200, response.json())
        else:
            return create_response(event, response.status_code, response.json())

def create_response(event, status_code, body):
    """Helper to structure the Bedrock Agent action group response."""
    return {
        'messageVersion': '1.0',
        'response': {
            'actionGroup': event['actionGroup'],
            'apiPath': event['apiPath'],
            'httpMethod': event['httpMethod'],
            'httpStatusCode': status_code,
            'responseBody': {
                'application/json': {
                    'body': json.dumps(body)
                }
            }
        }
    }
```

### 3. Handling the Agent's Pause and Resume State

When the Lambda returns a `401 Unauthorized` status code with the `auth_url`, the client-side application needs to intercept this and prompt the user.

Here is how your client-side orchestrator (e.g., a Node.js/Python backend talking to Bedrock) handles the response from the Agent Core:

```python
import boto3

bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')

def invoke_my_agent(user_id, user_prompt, session_id):
    response = bedrock_agent_runtime.invoke_agent(
        agentId="MY_AGENT_ID",
        agentAliasId="MY_ALIAS_ID",
        sessionId=session_id,
        inputText=user_prompt,
        sessionState={
            'promptSessionAttributes': {
                'userId': user_id
            }
        }
    )
    
    # Process the stream
    for event in response.get('completion', []):
        if 'chunk' in event:
            # Handle normal text output
            print(event['chunk']['bytes'].decode('utf-8'))
            
        elif 'returnControl' in event:
            # The agent paused execution because it requires client-side action (OAuth)
            invocation_id = event['returnControl']['invocationId']
            action_results = event['returnControl']['invocationInputs']
            
            # Extract the auth_url from the response
            # (In a real app, you would redirect the user to this URL)
            print(f"ACTION REQUIRED: Please authorize the application here: {auth_url}")
            
            # Once the user completes OAuth flow, resume agent execution using submit_action_results
            # bedrock_agent_runtime.submit_action_results(...)
```

---

## Security Best Practices for Agentic OAuth

Implementing the code is only half the battle. To ensure your AI agents do not become vectors for data breaches, adhere to these design paradigms:

1. **Ephemeral Scopes**: Request the absolute minimum scopes required. If your agent only needs to read calendar events, do not ask for full write access.
2. **Do Not Feed Tokens to the LLM**: Ensure your agent's system prompt and tools never ingest the actual access token string into the context window. The agent should only ever handle abstract references (e.g., "User is authorized") while the underlying Lambda securely fetches and uses the token.
3. **State Verification**: Always use the `state` parameter in OAuth authorization requests to prevent Cross-Site Request Forgery (CSRF). In agentic workflows, bind the `state` parameter to the unique agent execution session ID.
4. **Token Expiration & Refresh Handling**: Store refresh tokens securely in AWS Secrets Manager. Configure your action group Lambda to automatically refresh expired access tokens before attempting external API calls, preventing unnecessary auth redirects for the user.

---

## Conclusion: Trust is the Ultimate Feature

As we build increasingly complex AI agents, convenience will always pull developers toward taking security shortcuts. But one leaked token or unauthorized action can destroy user trust permanently.

By leveraging **Amazon Bedrock AgentCore** alongside a secure, decoupled OAuth consent loop, you can build autonomous agents that are both incredibly capable and enterprise-grade secure. 

Stop letting your agents run wild. Secure your workflows, respect user consent, and build for the future of trusted AI.