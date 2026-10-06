---
layout: post
title: "Your AI Agents Are Stealing User Data: How Amazon Bedrock AgentCore Finally Fixes the OAuth Nightmare"
date: 2026-07-15 16:01:58 +0530
excerpt: "Stop letting autonomous AI agents act as system superusers. Here is how to implement granular end-user OAuth consent using Amazon Bedrock AgentCore."
author: "Adarsh Nair"
categories: ai
tags: ["AWS", "Amazon Bedrock", "OAuth", "AI Agents", "Security"]
---

For the last two years, enterprise AI adoption has been quietly hiding a dirty secret. 

We built incredible autonomous agents capable of querying databases, filing Jira tickets, updating Salesforce records, and executing financial transactions. But under the hood, most of these deployments relied on a catastrophic security shortcut: **hardcoded service account credentials**.

When an enterprise user asks an LLM-powered agent to "summarize my pending Jira tasks and send a summary to my manager on Slack," the underlying agent traditionally executes those API calls using a single, high-privileged service token. The agent acts on behalf of the system, not the user. This exposes organizations to the notorious **Confused Deputy Problem**, scope creep, and catastrophic unauthorized data leakage.

If User A asks the agent to view restricted financial records, and the agent uses a master API key, nothing stops the LLM from leaking User B’s confidential data—other than fragile system prompt guardrails that can be bypassed with simple prompt injections.

To achieve enterprise-grade safety, AI agents must inherit the exact identity, permissions, and explicit consent of the end-user initiating the request. 

This is where **Amazon Bedrock AgentCore** comes into play. By embedding native 3-Legged OAuth (3LO) consent management directly into the agent orchestration layer, AgentCore allows developers to enforce granular user-level authorization without burning down their architecture.

In this deep dive, we will explore the security mechanics of Bedrock AgentCore, map out the authorization flow, and build a production-ready OAuth consent integration for an AI agent.

---

## The Core Problem: Static Tokens vs. Autonomous Non-Determinism

Traditional web applications use standard OAuth 2.0 flows: a user clicks "Log in with Google," consents to specific scopes (e.g., `read:email`), and the web backend receives an Access Token bound to that session. The execution path is predictable and hardcoded by software engineers.

AI agents break this mental model in two fundamental ways:

1. **Dynamic Execution Paths:** You cannot predict which APIs an LLM will decide to invoke at runtime. It might decide it needs Slack, Google Drive, and SAP in a single execution loop.
2. **Contextual Scope Elevation:** An agent might start a workflow needing read-only permissions (`drive.readonly`), but based on intermediate reasoning, decide it needs write permissions (`drive.file`) mid-conversation.

Without a dynamic authorization protocol built directly into the agent runtime, developers fall back on granting "super-tokens" to agents—giving an unpredictable, non-deterministic system full write access to core infrastructure.

```
[BAD APPROACH: Service Account Pattern]
User -> Prompt -> [ Bedrock Agent ] --(Master API Key)--> [ Third-Party API (Full Access!) ]

[GOOD APPROACH: AgentCore User-Delegated Identity Pattern]
User -> Prompt -> [ Bedrock Agent Core ] <--(OAuth 2.0 PKCE / Consent)--> [ End-User Identity Provider ]
                        |
                        +--(End-User OAuth Token)--> [ Third-Party API (Scoped Access) ]
```

---

## Architectural Overview: Amazon Bedrock AgentCore OAuth Flow

Amazon Bedrock AgentCore introduces a managed Identity Execution Boundary. Instead of handling token storage, token refresh loops, and user redirection within your custom orchestration code or Lambda functions, AgentCore manages the lifecycle of 3-Legged OAuth connections.

Here is how the end-user OAuth consent flow operates under the hood when an agent encounters a tool requiring authorization:

```
+----------+          +-----------------------+          +------------------------+          +--------------------+
| End User |          | Bedrock AgentRuntime  |          | AgentCore Auth Manager |          | OAuth 2.0 Provider |
+----+-----+          +-----------+-----------+          +-----------+------------+          +---------+----------+
     |                            |                                  |                             |
     | 1. "Sync my Jira tasks"    |                                  |                             |
     +--------------------------->|                                  |                             |
     |                            | 2. Intercept Action Group Call   |                             |
     |                            +--------------------------------->|                             |
     |                            |                                  | 3. Check Token Cache        |
     |                            |                                  | (Token Missing/Expired)     |
     |                            |                                  |                             |
     |                            | 4. Return Authorization Required |                             |
     |                            |<---------------------------------+                             |
     |                            | (Contains Redirect URI & State)  |                             |
     |                            |                                  |                             |
     | 5. Render Consent Link     |                                  |                             |
     |<---------------------------+                                  |                             |
     |                            |                                  |                             |
     | 6. Authenticate & Approve Consent Scope                       |                             |
     +-------------------------------------------------------------------------------------------->|
     |                                                                                             |
     | 7. OAuth Redirect with Auth Code                                                            |
     +-------------------------------------------------------------------------------------------->|
     |                                                                                             | 8. Exchange Code for
     |                                                                                             |    User Access Token
     |                                                                                             +------------------->
     |                            | 9. Token Injected into Context   |                             |
     |                            |<---------------------------------+                             |
     |                            |                                                                |
     |                            | 10. Execute Tool Action using User Access Token                |
     |                            +--------------------------------------------------------------->|
```

### Key Phases:
1. **Tool Discovery & Interception:** The LLM decides to call a tool defined in an Action Group. AgentCore inspects the security scheme of the OpenAPI tool definition.
2. **State & Challenge Generation:** If no valid OAuth token exists for the active `end_user_id` session, AgentCore halts agent execution, generates a Proof Key for Code Exchange (PKCE) challenge, and constructs a secure authorization URL.
3. **Out-of-Band Authorization:** The client UI intercepts the Agent's return status, prompts the user to authenticate with the Identity Provider (IdP), and collects consent.
4. **Token Injection:** Once authenticated, the IdP sends the authorization code back to AgentCore's callback endpoint. AgentCore securely exchanges it for an Access/Refresh token pair, stores it in an encrypted session vault, and resumes agent execution.

---

## Step-by-Step Implementation

Let's build an Amazon Bedrock Agent action group that executes actions on behalf of a user using **Amazon Bedrock AgentCore** integrated with an OAuth 2.0 provider (such as Amazon Cognito, Auth0, or Okta).

### Step 1: Define the Tool OpenAPI Specification with OAuth Security Scheme

First, we need to create an OpenAPI 3.0 specification for our action group tool. Notice how we define the `securitySchemes` to enforce OAuth 2.0 `authorizationCode` flow directly inside the schema.

```yaml
# jira_action_group.yaml
openapi: 3.0.1
info:
  title: Jira Integration Action Group
  version: 1.0.0
paths:
  /issue/create:
    post:
      summary: Create a new Jira issue on behalf of the user
      operationId: createJiraIssue
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                project_key:
                  type: string
                  description: The key of the Jira project
                summary:
                  type: string
                  description: The summary title of the issue
                description:
                  type: string
                  description: Detailed description of the task
              required: [project_key, summary]
      responses:
        '200':
          description: Issue created successfully
      security:
        - UserOAuth2:
            - read:jira-work
            - write:jira-work

components:
  securitySchemes:
    UserOAuth2:
      type: oauth2
      description: Amazon Bedrock AgentCore End-User OAuth Delegation
      flows:
        authorizationCode:
          authorizationUrl: https://auth.yourdomain.com/oauth2/authorize
          tokenUrl: https://auth.yourdomain.com/oauth2/token
          refreshUrl: https://auth.yourdomain.com/oauth2/token
          scopes:
            read:jira-work: Grants permission to read user Jira issues
            write:jira-work: Grants permission to create and edit Jira issues
```

---

### Step 2: Configure the AgentCore Authorization Policy

Next, we establish the AgentCore consent governance policy using the AWS SDK / JSON configuration. This JSON snippet dictates how AgentCore enforces user consent, session isolation, and token lifespan limits.

```json
{
  "Version": "2026-09-15",
  "AgentCoreConfig": {
    "AgentId": "AGENT_CORE_JIRA_ASSISTANT_01",
    "AuthStrategy": "END_USER_DELEGATED_OAUTH",
    "OAuthConfiguration": {
      "IdentityProviderName": "Enterprise-Auth0-IdP",
      "GrantType": "AUTHORIZATION_CODE_PKCE",
      "TokenStore": {
        "KmsKeyArn": "arn:aws:kms:us-east-1:123456789012:key/a1b2c3d4-e5f6-7890-abcd-1234567890ab",
        "EncryptedStorageTier": "HARDENED_SESSION_VAULT"
      },
      "ConsentManagement": {
        "PromptBehavior": "CONSENT_IF_SCOPE_ELEVATED",
        "TokenMaxLifetimeSeconds": 3600,
        "AllowAutoRefresh": true
      }
    }
  }
}
```

---

### Step 3: Implement the Auth-Aware Lambda Target Handler

When AgentCore verifies that valid user tokens exist in the session context, it injects the claims and access token directly into the Lambda event payload within `event['requestBody']['authorizationContext']`.

Here is the Python (Boto3/Lambda) backend that securely consumes the user-delegated token to call the upstream Jira REST API:

```python
import json
import os
import urllib3

http = urllib3.PoolManager()

def lambda_handler(event, context):
    print("Received Bedrock Agent Event:", json.dumps(event))
    
    # 1. Extract Identity and Context from AgentCore Authorization Envelope
    auth_context = event.get('authorizationContext', {})
    user_access_token = auth_context.get('userAccessToken')
    end_user_id = auth_context.get('endUserId')
    
    # Guardrail check: Enforce that user token exists
    if not user_access_token:
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event['actionGroup'],
                "apiPath": event['apiPath'],
                "httpMethod": event['httpMethod'],
                "httpStatusCode": 401,
                "responseBody": {
                    "application/json": {
                        "body": json.dumps({"error": "User OAuth Consent required to perform this action."})
                    }
                }
            },
            "sessionAttributes": {
                "auth_status": "REQUIRES_USER_CONSENT"
            }
        }
    
    # 2. Extract Agent Function Parameters
    api_path = event.get('apiPath')
    properties = event.get('requestBody', {}).get('content', {}).get('application/json', {}).get('properties', [])
    
    param_dict = {p['name']: p['value'] for p in properties}
    
    if api_path == "/issue/create":
        jira_url = f"https://your-domain.atlassian.net/rest/api/3/issue"
        
        payload = {
            "fields": {
                "project": {"key": param_dict.get("project_key")},
                "summary": param_dict.get("summary"),
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [{
                        "type": "paragraph",
                        "content": [{
                            "text": f"{param_dict.get('description', '')}\n\n[Created via Bedrock Agent by User ID: {end_user_id}]",
                            "type": "text"
                        }]
                    }]
                },
                "issuetype": {"name": "Task"}
            }
        }
        
        # 3. Call Upstream API using the End-User's Delegated Bearer Token!
        headers = {
            "Authorization": f"Bearer {user_access_token}",
            "Content-Type": "application/json"
        }
        
        response = http.request(
            "POST",
            jira_url,
            body=json.dumps(payload),
            headers=headers
        )
        
        jira_response = json.loads(response.data.decode('utf-8'))
        
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event['actionGroup'],
                "apiPath": event['apiPath'],
                "httpMethod": event['httpMethod'],
                "httpStatusCode": response.status,
                "responseBody": {
                    "application/json": {
                        "body": json.dumps({
                            "issue_id": jira_response.get("id"),
                            "key": jira_response.get("key"),
                            "status": "Created"
                        })
                    }
                }
            }
        }

    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event['actionGroup'],
            "httpStatusCode": 400,
            "responseBody": {"application/json": {"body": json.dumps({"error": "Unsupported API Path"})}}
        }
    }
```

---

## Operational Best Practices & Common Pitfalls

Implementing end-user consent for autonomous systems introduces operational friction if not architected cleanly. Below are critical production considerations:

### 1. Guarding Against "Consent Fatigue"
If your agent requests individual consent for every minor execution step, users will blindly click "Accept" without reading scope boundaries—defeating the entire purpose of security.
* **Solution:** Implement **Progressive Consent**. Define baseline read-only scopes during the initial user session onboarding. Only trigger step-up authentication prompts when the agent transitions to critical mutation operations (e.g., executing transactions, deleting resources, modifying access controls).

### 2. Token Revocation and Cache Invalidation
What happens when a user revokes app permissions directly inside Atlassian or Google Settings, but AgentCore still holds a cached active session token?
* **Solution:** Configure AgentCore's Token Validation policy to execute lightweight token introspections (`/oauth2/introspect`) before invoking long-running agent execution workflows.

### 3. Agent Session Isolation
Never allow an agent session key to be shared across multi-tenant user threads. AgentCore locks session memory to an explicit cryptographic pairing of `(SessionId + EndUserId)`. Ensure your front-end web application enforces this pairing when calling Bedrock's `InvokeAgent` API:

```javascript
// Front-End SDK Call Example
const response = await bedrockAgentRuntimeClient.send(new InvokeAgentCommand({
  agentId: "AGENT_CORE_JIRA_ASSISTANT_01",
  agentAliasId: "TSTALIASID",
  sessionId: userSessionId, // Bound to authenticated browser session
  endUserId: "usr_9982310_enterprise", // Cryptographically verified JWT sub claim
  inputText: "Create a bug ticket in project SEC for missing OAuth scopes"
}));
```

---

## Conclusion: Securing the Autonomous Frontier

The era of trusting AI agents with master enterprise keys is over. As agents evolve from simple chat interfaces to fully autonomous workers capable of triggering business workflows across fragmented SaaS landscapes, identity must form the perimeter.

Amazon Bedrock AgentCore provides the missing bridge between modern OAuth 2.0 authorization frameworks and non-deterministic agentic workflows. By placing identity delegation directly into the agent runtime, developers can build tools that execute with maximum autonomy while strictly respecting human authority.

### Deployment Checklist for Security Teams:
- [ ] Audit all Bedrock Action Groups to eliminate hardcoded API keys/service accounts.
- [ ] Implement OpenAPI 3.0 `securitySchemes` across all tool definitions.
- [ ] Enable KMS encryption on AgentCore session vaults.
- [ ] Enforce dynamic PKCE authorization flows for web/mobile client interfaces.
- [ ] Establish strict dynamic scope limits to enforce the Principle of Least Privilege.

*How is your engineering team handling identity propagation in autonomous agent architectures? Reach out or leave a comment below!*