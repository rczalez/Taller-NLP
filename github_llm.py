import os
import requests
from dotenv import load_dotenv

load_dotenv()

class GitHubModelsError(Exception):
    pass


def chat_completion(
    messages,
    model: str | None = None,
    endpoint: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise GitHubModelsError("GITHUB_TOKEN not found in environment")

    endpoint = endpoint or os.getenv(
        "GITHUB_MODELS_ENDPOINT",
        "https://models.inference.ai.azure.com/chat/completions",
    )

    model = model or os.getenv("MODEL_NAME")
    if not model:
        raise GitHubModelsError("MODEL_NAME not set")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    r = requests.post(endpoint, headers=headers, json=payload, timeout=60)

    if r.status_code != 200:
        raise GitHubModelsError(
            f"GitHub Models API error {r.status_code}: {r.text}"
        )

    data = r.json()

    # THIS is the key line you were missing:
    return data["choices"][0]["message"]["content"]
