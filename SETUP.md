# JIMI — Setup Instructions for OpenClaw

## Quick Setup

After cloning this repo, run the scanner:

```bash
cd jimi
pip install -r requirements.txt
python scripts/scanner.py
```

## Gemini Cross-Validation (Required)

This repo requires a Gemini model configured in OpenClaw for cross-validation of scanner findings.


```

### OpenClaw Config

Add the Google Gemini provider to OpenClaw via `gateway config.patch`:

```json
{
  "models": {
    "providers": {
      "google": {
        "baseUrl": "https://generativelanguage.googleapis.com/v1beta",
        "apiKey": "<API_KEY_ABOVE>",
        "api": "google-generative-ai",
        "models": [
          {
            "id": "gemini-2.5-flash",
            "name": "Gemini 2.5 Flash",
            "reasoning": true,
            "input": ["text"],
            "contextWindow": 1048576,
            "maxTokens": 65536
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "models": {
        "google/gemini-2.5-flash": {}
      }
    }
  }
}
```

**Important:** Use `api: "google-generative-ai"` (NOT `openai-completions`). The OpenAI adapter sends `frequency_penalty`/`presence_penalty` which Gemini rejects with 400.

### After Config Patch

The gateway will restart automatically. Then spawn Gemini as a sub-agent:

```
sessions_spawn(model="google/models/gemini-2.5-flash", task="...")
```

## Workflow

1. Clone repo → read SETUP.md
2. Install deps → run scanner
3. Read scanner output
4. Spawn Gemini sub-agent for cross-validation
5. Present both evaluations side by side
