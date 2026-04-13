"""
ViralClip AI - Gemini Provider (Fallback)
Uses Gemini Flash via Google AI Studio for handling large contexts or as a fallback.
Migrated to google.genai (new SDK) — google.generativeai is deprecated.
"""
import json
from typing import List

from google import genai
from google.genai import types

from src.models import FullTranscript, ViralClip
from src.config import config
from . import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model_name = config.GEMINI_MODEL

    @property
    def provider_name(self) -> str:
        return f"Gemini ({self.model_name})"

    def analyze_virality(self, transcript: FullTranscript, num_clips: int) -> List[ViralClip]:
        prompt = self.build_prompt(transcript, num_clips)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7,
            ),
        )

        raw_json = response.text

        try:
            data = json.loads(raw_json)
            if isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, list):
                        data = val
                        break

            clips = [ViralClip(**item) for item in data]
            return clips

        except Exception as e:
            raise RuntimeError(f"Failed to parse Gemini output: {e}\nRaw: {raw_json}")
