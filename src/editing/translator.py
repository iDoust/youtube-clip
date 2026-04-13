"""
Translator Module
Handles batch translation of video subtitles/transcripts using LLMs (OpenRouter/Gemini).
"""
import json
from typing import List, Dict

from openai import OpenAI
import google.generativeai as genai

from src.config import config


class SubtitleTranslator:
    def __init__(self):
        # Primary: OpenRouter, Fallback: Gemini
        self.openrouter_client = None
        self.gemini_model = None

        if config.OPENROUTER_API_KEY:
            self.openrouter_client = OpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
            )

        if config.GEMINI_API_KEY:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(model_name=config.GEMINI_MODEL)

    def _chunk_list(self, lst: List, n: int) -> List[List]:
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _build_translation_prompt(self, segments: List[Dict], target_lang: str) -> str:
        return (
            f"Translate the following subtitle segments into {target_lang}.\n"
            "Maintain the exact same JSON array structure, only modifying the 'text' fields.\n"
            "Output ONLY a valid JSON array. Do NOT wrap the output in markdown blocks.\n\n"
            f"Input:\n{json.dumps(segments, ensure_ascii=False)}"
        )

    def _parse_json_response(self, raw_text: str) -> List[Dict]:
        """Strip markdown fences if present and parse JSON."""
        raw = raw_text.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return json.loads(raw.strip())

    def translate_batch_openrouter(self, segments: List[Dict], target_lang: str):
        """Translates a batch of segments using OpenRouter."""
        if not self.openrouter_client:
            return None

        prompt = self._build_translation_prompt(segments, target_lang)

        try:
            response = self.openrouter_client.chat.completions.create(
                model=config.OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert subtitle translator. Output ONLY valid JSON arrays."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                extra_headers={
                    "HTTP-Referer": "https://github.com/viralclip-ai",
                    "X-Title": "ViralClip AI",
                },
            )
            return self._parse_json_response(response.choices[0].message.content)
        except Exception as e:
            print(f"[-] OpenRouter translation failed: {e}")
            return None

    def translate_batch_gemini(self, segments: List[Dict], target_lang: str):
        """Translates a batch of segments using Gemini."""
        if not self.gemini_model:
            return None

        prompt = self._build_translation_prompt(segments, target_lang)

        try:
            response = self.gemini_model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"[-] Gemini translation failed: {e}")
            return None

    def translate_subtitles(self, segments: List[Dict], target_lang: str = "id") -> List[Dict]:
        """
        Takes a list of transcript dicts [{start, end, text}...]
        and translates the text field in batches of 20.
        """
        print(f"[*] Translating {len(segments)} segments to '{target_lang}'...")
        translated_segments = []

        batches = list(self._chunk_list(segments, 20))

        for i, batch in enumerate(batches):
            print(f"    - Translating batch {i+1}/{len(batches)}...")

            result = self.translate_batch_openrouter(batch, target_lang)

            if result is None:
                print("    - OpenRouter failed, falling back to Gemini...")
                result = self.translate_batch_gemini(batch, target_lang)

            if result is None:
                print("    - All providers failed. Keeping original text.")
                translated_segments.extend(batch)
            else:
                # Ensure timestamps match original in case LLM hallucinates
                for orig, trans in zip(batch, result):
                    trans['start'] = orig['start']
                    trans['end'] = orig['end']
                translated_segments.extend(result)

        return translated_segments
