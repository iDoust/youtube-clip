"""
ViralClip AI - OpenRouter Provider (Primary)
Uses OpenRouter API (OpenAI-compatible) to access hundreds of models
via a single API key: https://openrouter.ai
"""
import json
from typing import List

from openai import OpenAI

from src.models import FullTranscript, ViralClip
from src.config import config
from . import LLMProvider


class OpenRouterProvider(LLMProvider):
    def __init__(self):
        if not config.LLM_API_KEY:
            raise ValueError("LLM_API_KEY not found in environment.")

        self.client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
        )
        self.model = config.LLM_MODEL

    def provider_name(self) -> str:
        # Determine a generic provider name based on the base_url
        provider = "OpenAI-API"
        if "openrouter" in config.LLM_BASE_URL:
            provider = "OpenRouter"
        elif "minimax" in config.LLM_BASE_URL:
            provider = "MiniMax"
            
        return f"{provider} ({self.model})"

    def analyze_virality(self, transcript: FullTranscript, num_clips: int) -> List[ViralClip]:
        prompt = self.build_prompt(transcript, num_clips)

        # Some providers like generic OpenAI/Minimax might not support extra_headers gracefully,
        # so we only include them if it's openrouter
        extra_kwargs = {}
        if "openrouter" in config.LLM_BASE_URL:
            extra_kwargs["extra_headers"] = {
                "HTTP-Referer": "https://github.com/viralclip-ai",
                "X-Title": "ViralClip AI",
            }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
            **extra_kwargs
        )

        raw_json = response.choices[0].message.content

        # Extract JSON if there are <think> blocks or markdown codeblocks
        import re
        if "```json" in raw_json:
            match = re.search(r'```json\s*(.*?)\s*```', raw_json, re.DOTALL)
            if match:
                raw_json = match.group(1)
        elif raw_json.strip().startswith("<think>"):
            # If it starts with <think>, discard everything up to </think>
            match = re.search(r'</think>\s*(.*)', raw_json, re.DOTALL)
            if match:
                raw_json = match.group(1)
                
        # Kadang model merespon dengan array flat di luar backticks
        if not raw_json.strip().startswith("{") and not raw_json.strip().startswith("["):
             match = re.search(r'(\[.*\]|\{.*\})', raw_json, re.DOTALL)
             if match:
                 raw_json = match.group(1)

        try:
            data = json.loads(raw_json)
            if isinstance(data, dict):
                # Unwrap {clips: [...]} or similar wrapper
                for val in data.values():
                    if isinstance(val, list):
                        data = val
                        break
                else:
                    data = list(data.values())[0]

            clips = [ViralClip(**item) for item in data]
            return clips

        except Exception as e:
            raise RuntimeError(f"Failed to parse OpenRouter output: {e}\nRaw: {raw_json}")
