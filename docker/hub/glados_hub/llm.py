"""
LLM clients for the hub:

  * Gemini (Google)  — primary source for GLaDOS's *responses*
  * local LLM (radiance, qwen3.8-27b; formerly llama.cpp/Gemma) —
                       the *tiny* LLM used for YES/NO gatekeeping,
                       and the fallback if Gemini is unavailable.

Request/response builders are pure functions (unit-testable); the async
callers wrap them with httpx.
"""

from __future__ import annotations

import logging
import os
import re

import httpx

logger = logging.getLogger("glados_hub.llm")

# Chat-template tokens that can leak out of local LLM outputs.
_CHAT_TOKEN_RE = re.compile(
    r"(<\|[^>|]*\|>|<\|im[^>]*>|</?s>|<eot>|<\|end_of_text\|>|\[\[.*?\]\])",
    re.IGNORECASE,
)


def clean_llm_output(text: str) -> str:
    """Strip leaked chat-template tokens and collapse blank lines."""
    if not text:
        return ""
    text = _CHAT_TOKEN_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_for_tts(text: str) -> str:
    """Convert LLM text into something a TTS engine can read naturally.

    Strips markdown/code markup, turns non-speakable punctuation into
    speech pauses, and normalizes units (72°F -> '72 degrees Fahrenheit').
    """
    if not text:
        return ""
    t = text
    # Code blocks and inline code
    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    # Markdown structure
    t = re.sub(r"^\s*[-*•]\s+", "", t, flags=re.MULTILINE)  # bullet lists
    t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.MULTILINE)  # numbered lists
    t = re.sub(r"#{1,6}\s*", "", t)                          # headers
    t = re.sub(r"(\*\*|__)(.+?)\1", r"\2", t)                # bold
    t = re.sub(r"(?<!\w)(\*|_)(?!\s)(.+?)(?<!\s)\1(?!\w)", r"\2", t)  # italic
    # Images and links
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"https?://\S+", " ", t)
    # Units and symbols -> spoken form
    t = re.sub(r"°\s*F", " degrees Fahrenheit", t, flags=re.IGNORECASE)
    t = re.sub(r"°\s*C", " degrees Celsius", t, flags=re.IGNORECASE)
    t = re.sub(r"°", " degrees", t)
    t = t.replace("&", " and ")
    t = re.sub(r"[—–]", ", ", t)
    # Punctuation: colons/semicolons read as pauses, not words
    t = re.sub(r"\s*:\s*", ", ", t)
    t = t.replace(";", ", ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

GEMINI_API_BASE = os.environ.get(
    "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/models"
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
# Local LLM endpoint. Was the llama.cpp server (glados_llama, Gemma 4 E4B);
# now the radiance engine on the host (qwen3.8-27b). LLAMA_URL is kept as a
# fallback so the old service can be rolled back in one step.
LLM_URL = os.environ.get("LLM_URL") or os.environ.get(
    "LLAMA_URL", "http://glados_llama:8080"
)
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.8-27b")

FALLBACK_LINE = "I'm sorry, but I'm currently undergoing testing. Please stand by."


def _to_gemini_contents(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-style messages to Gemini 'contents'.

    System messages are excluded (they go into systemInstruction).
    Consecutive same-role messages are merged (Gemini requires
    alternating roles).
    """
    contents: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        role = "model" if msg.get("role") == "assistant" else "user"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append({"text": msg["content"]})
        else:
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    return contents


def build_gemini_request(
    messages: list[dict],
    system_prompt: str,
    max_output_tokens: int = 300,
    temperature: float = 0.7,
) -> dict:
    return {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": _to_gemini_contents(messages),
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
        },
    }


def gemini_url(model: str = GEMINI_MODEL) -> str:
    return f"{GEMINI_API_BASE}/{model}:generateContent"


def parse_gemini_response(data: dict) -> str:
    """Extract the text from a Gemini generateContent response."""
    for candidate in data.get("candidates", []):
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            if part.get("text"):
                return part["text"].strip()
    return ""


async def call_gemini(
    client: httpx.AsyncClient,
    messages: list[dict],
    system_prompt: str,
    max_output_tokens: int = 300,
    temperature: float = 0.7,
) -> str:
    """Call Gemini generateContent. Raises on transport/API failure."""
    url = gemini_url()
    headers = {"x-goog-api-key": GEMINI_API_KEY}
    resp = await client.post(url, json=build_gemini_request(
        messages, system_prompt, max_output_tokens, temperature
    ), headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:200]}")
    text = clean_llm_output(parse_gemini_response(resp.json()))
    if not text:
        raise RuntimeError("Gemini returned no text")
    return text


def build_llama_request(
    messages: list[dict],
    max_tokens: int = 300,
    temperature: float = 0.7,
    model: str = LLM_MODEL,
    tools: list[dict] | None = None,
) -> dict:
    """OpenAI-compatible body for the local LLM (radiance, ex-llama.cpp).

    Thinking is disabled per request: GLaDOS calls are short utility turns
    (the YES/NO gate runs on an 8-token budget, replies cap at 300) and
    reasoning tokens would consume the whole budget before any answer.
    Radiance lets a request override the server default, so other consumers
    of the same model (Open WebUI, Cline) keep thinking enabled.
    """
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        body["tools"] = tools
    return body


def llama_headers() -> dict:
    """Auth for the local LLM. Radiance requires an API key on every request
    but /health, /ping and the dashboard; a llama.cpp server ignores the
    header, which is what makes the one-step rollback work."""
    key = os.environ.get("LLM_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


async def call_llama_message(
    client: httpx.AsyncClient,
    messages: list[dict],
    max_tokens: int = 300,
    temperature: float = 0.7,
    model: str = LLM_MODEL,
    tools: list[dict] | None = None,
) -> dict:
    """Call the local LLM and return the full message dict (content + tool_calls).

    Tool-call traffic flows through this: the caller appends the assistant
    message, executes the calls, feeds 'tool' messages back, and repeats
    until a content-only message arrives.
    """
    resp = await client.post(
        f"{LLM_URL}/v1/chat/completions",
        json=build_llama_request(messages, max_tokens, temperature, model, tools=tools),
        headers=llama_headers(),
    )
    data = resp.json()
    msg = data["choices"][0]["message"]
    if msg.get("content"):
        msg["content"] = clean_llm_output(msg["content"])
    return msg


async def call_llama(
    client: httpx.AsyncClient,
    messages: list[dict],
    max_tokens: int = 300,
    temperature: float = 0.7,
    model: str = LLM_MODEL,
) -> str:
    """Call the local LLM's OpenAI-compatible endpoint (text content only)."""
    msg = await call_llama_message(client, messages, max_tokens, temperature, model)
    return msg.get("content") or ""