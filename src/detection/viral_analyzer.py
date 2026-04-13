"""
ViralClip AI - LLM Orchestrator
Manages the fallback chain: OpenRouter → Gemini.
Post-validates clip boundaries to ensure complete context.
"""
from typing import List

from src.models import FullTranscript, ViralClip, TranscriptSegment
from src.llm_providers.openrouter_provider import OpenRouterProvider
from src.llm_providers.gemini_provider import GeminiProvider


class ViralAnalyzer:
    def __init__(self):
        # Fallback chain: OpenRouter (primary) → Gemini (fallback)
        providers = []

        try:
            providers.append(OpenRouterProvider())
        except ValueError:
            pass  # Key not set, skip

        try:
            providers.append(GeminiProvider())
        except Exception:
            pass

        if not providers:
            raise RuntimeError(
                "No LLM provider configured. "
                "Set OPENROUTER_API_KEY or GEMINI_API_KEY in .env"
            )

        self.providers = providers

    def analyze(self, transcript: FullTranscript, num_clips: int = 3) -> List[ViralClip]:
        """
        Attempts to analyze the transcript using the provider chain.
        Post-validates boundaries to prevent mid-sentence cutoffs.
        """
        last_error = None

        for provider in self.providers:
            try:
                print(f"[*] Attempting analysis with {provider.provider_name}...")
                clips = provider.analyze_virality(transcript, num_clips)
                print(f"[+] Successfully generated {len(clips)} clips with {provider.provider_name}.")

                # Post-validate: fix any clips that cut mid-sentence
                clips = [self._validate_clip_boundaries(clip, transcript.segments) for clip in clips]

                return sorted(clips, key=lambda x: x.virality_score, reverse=True)

            except Exception as e:
                print(f"[-] Fast fallback triggered. {provider.provider_name} failed: {e}")
                last_error = e
                continue  # Try the next provider

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def _validate_clip_boundaries(self, clip: ViralClip, segments: List[TranscriptSegment]) -> ViralClip:
        """
        Programmatically fix clip boundaries so they always land on complete sentences.
        - If end_time cuts mid-sentence → extend to next sentence end
        - If start_time cuts mid-sentence → pull back to previous sentence start
        Max extension: 15 seconds (prevent runaway extensions)
        """
        original_start = clip.start_time
        original_end = clip.end_time

        # --- Fix END boundary ---
        end_seg = None
        for seg in segments:
            if seg.start <= clip.end_time <= seg.end:
                end_seg = seg
                break
            if seg.start > clip.end_time:
                break

        if end_seg and not end_seg.text.strip().rstrip().endswith(('.', '!', '?', '。')):
            for seg in segments:
                if seg.start < clip.end_time:
                    continue
                text = seg.text.strip()
                if text.endswith(('.', '!', '?', '。')) or (seg.start - clip.end_time) > 2.0:
                    new_end = seg.end
                    if new_end - original_end <= 15.0:
                        clip.end_time = new_end
                    break

        # --- Fix START boundary ---
        start_seg = None
        for seg in segments:
            if seg.start <= clip.start_time <= seg.end:
                start_seg = seg
                break

        if start_seg and start_seg.start < clip.start_time:
            if clip.start_time - start_seg.start <= 5.0:
                clip.start_time = start_seg.start

        prev_seg = None
        for seg in segments:
            if seg.end <= clip.start_time:
                prev_seg = seg
            else:
                break

        if prev_seg and not prev_seg.text.strip().endswith(('.', '!', '?', '。')):
            if clip.start_time - prev_seg.start <= 10.0:
                clip.start_time = prev_seg.start

        # Log adjustments
        if clip.start_time != original_start or clip.end_time != original_end:
            delta_s = original_start - clip.start_time
            delta_e = clip.end_time - original_end
            adj = []
            if delta_s > 0.5:
                adj.append(f"start -{delta_s:.0f}s")
            if delta_e > 0.5:
                adj.append(f"end +{delta_e:.0f}s")
            if adj:
                print(f"    [~] Boundary fix: {', '.join(adj)} (context completion)")

        return clip
