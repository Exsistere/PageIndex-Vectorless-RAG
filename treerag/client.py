"""
OpenAI API client wrapper for TreeRAG.
Handles retries, JSON parsing, and prompt management.
"""

from __future__ import annotations
import json
import os
import time
import logging
from typing import Optional
from .utils import track_time, track_llm_call

logger = logging.getLogger("treerag")


class OpenAIClient:
    """Thin wrapper around the OpenAI API with retry and JSON helpers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("CHATGPT_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key="
            )
        self._client = OpenAI(api_key=key)

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 4096, cache_key: Optional[str] = None) -> str:
        """Send a chat completion request and return the response text."""
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    prompt_cache_key= cache_key
                )
                # logger.info(f"{resp.usage}")
                return {
                    "content" : resp.choices[0].message.content or "",
                    "usage": resp.usage
                }
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"OpenAI call failed (attempt {attempt+1}): {e}. Retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def chat_json(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        cache_key: Optional[str] = None,
    ) -> dict | list:
        """
        Send a chat completion and parse the response as JSON.
        Strips markdown code fences if present.
        """
        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens, cache_key = cache_key)
        return {
            "content" : self._parse_json(raw["content"]),
            "usage": raw["usage"]
        }
    def _parse_json(self, text: str) -> dict | list:
        # Strip ```json ... ``` fences
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON substring
            start = text.find("{") if "{" in text else text.find("[")
            end = text.rfind("}") if "{" in text else text.rfind("]")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end+1])
                except Exception:
                    pass
            raise ValueError(f"Failed to parse JSON from response:\n{text[:500]}")
