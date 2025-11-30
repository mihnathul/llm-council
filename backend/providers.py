"""Multi-provider LLM client for direct API calls to OpenAI, Google, and Anthropic."""

import httpx
from typing import List, Dict, Any, Optional
from .config import OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY


def get_provider(model: str) -> str:
    """Determine the provider from the model identifier."""
    if model.startswith("openai/"):
        return "openai"
    elif model.startswith("google/"):
        return "google"
    elif model.startswith("anthropic/"):
        return "anthropic"
    else:
        raise ValueError(f"Unknown provider for model: {model}")


def get_model_name(model: str) -> str:
    """Extract the model name from the identifier (e.g., 'openai/gpt-4o' -> 'gpt-4o')."""
    return model.split("/", 1)[1]


async def query_openai(
    model_name: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """Query OpenAI API directly."""
    if not OPENAI_API_KEY:
        print(f"Error: OPENAI_API_KEY not set")
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }
    except Exception as e:
        print(f"Error querying OpenAI model {model_name}: {e}")
        return None


async def query_google(
    model_name: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """Query Google Gemini API directly."""
    if not GOOGLE_API_KEY:
        print(f"Error: GOOGLE_API_KEY not set")
        return None

    # Convert messages to Gemini format
    contents = []
    system_instruction = None

    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        else:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": 8192,
        }
    }

    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GOOGLE_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            content = data['candidates'][0]['content']['parts'][0]['text']

            return {
                'content': content,
                'reasoning_details': None
            }
    except Exception as e:
        print(f"Error querying Google model {model_name}: {e}")
        return None


async def query_anthropic(
    model_name: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """Query Anthropic Claude API directly."""
    if not ANTHROPIC_API_KEY:
        print(f"Error: ANTHROPIC_API_KEY not set")
        return None

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    # Extract system message if present
    system_content = None
    filtered_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            filtered_messages.append(msg)

    payload = {
        "model": model_name,
        "max_tokens": 8192,
        "messages": filtered_messages,
    }

    if system_content:
        payload["system"] = system_content

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            content = data['content'][0]['text']

            return {
                'content': content,
                'reasoning_details': None
            }
    except Exception as e:
        print(f"Error querying Anthropic model {model_name}: {e}")
        return None


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a model via its native API.

    Args:
        model: Model identifier (e.g., "openai/gpt-4o", "google/gemini-2.0-flash", "anthropic/claude-sonnet-4-20250514")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    provider = get_provider(model)
    model_name = get_model_name(model)

    if provider == "openai":
        return await query_openai(model_name, messages, timeout)
    elif provider == "google":
        return await query_google(model_name, messages, timeout)
    elif provider == "anthropic":
        return await query_anthropic(model_name, messages, timeout)
    else:
        print(f"Unknown provider: {provider}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    tasks = [query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)

    return {model: response for model, response in zip(models, responses)}
