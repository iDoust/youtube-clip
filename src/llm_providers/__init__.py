"""
ViralClip AI - Base LLM Provider
Defines the interface all providers must follow + shared prompt builder.
"""
from abc import ABC, abstractmethod
from typing import List
from src.models import FullTranscript, ViralClip


class LLMProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the LLM provider."""
        pass

    @abstractmethod
    def analyze_virality(self, transcript: FullTranscript, num_clips: int) -> List[ViralClip]:
        """
        Analyzes the transcript and returns a list of potential viral clips.
        Raises an exception if the API call fails or rate limits are hit.
        """
        pass

    @staticmethod
    def build_prompt(transcript: FullTranscript, num_clips: int) -> str:
        """
        Shared prompt for all providers. Single source of truth.
        Presents transcript as flowing text with periodic timestamps
        so the LLM can understand topic flow naturally.
        """
        # Build flowing transcript with timestamp markers every ~30 seconds
        paragraphs = []
        current_paragraph = []
        last_marker_time = -30  # Force first marker

        for seg in transcript.segments:
            # Insert timestamp marker every ~30 seconds
            if seg.start - last_marker_time >= 30:
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
                mins = int(seg.start // 60)
                secs = int(seg.start % 60)
                current_paragraph.append(f"\n[{mins:02d}:{secs:02d}]")
                last_marker_time = seg.start

            current_paragraph.append(seg.text.strip())

        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

        transcript_text = "\n".join(paragraphs)

        return f"""You are an expert content strategist. Analyze this video transcript and find the top {num_clips} viral clip segments.

=== SKIP ZONE ===
NEVER select clips from the first 120 seconds ([00:00]-[02:00]) of the video.
The first two minutes is the original video's hook/intro — we want the actual SUBSTANCE, not the opening.
Start looking from [02:00] onwards.

=== HOW TO READ THE TRANSCRIPT ===
The transcript below is flowing text with timestamp markers like [MM:SS] every ~30 seconds.
Read it like a conversation. Understand the FULL topic flow before selecting clip boundaries.

=== CRITICAL: COMPLETE CONTEXT — READ THIS CAREFULLY ===
The #1 mistake is cutting a clip BEFORE the topic finishes. To avoid this:

1. First, READ the entire transcript and identify where each TOPIC starts and ends
2. A topic ends when: the speaker changes subject, there's a conclusion/punchline, or a natural pause
3. Only THEN select start_time and end_time that capture the COMPLETE topic

RULES:
- NEVER cut mid-sentence or mid-story
- If a speaker says "Tips pertama..." you MUST include all the tips until they finish
- If a speaker tells a story, you MUST include the conclusion/punchline
- If a speaker lists items, you MUST include ALL items in the list
- Add 5-second BUFFER: start 5s before the hook, end 5s after the conclusion
- A complete 90-second clip is ALWAYS better than an incomplete 45-second clip

BAD example: Clip ends at "jadi menurut saya..." (cut mid-thought!)  
GOOD example: Clip ends at "...itulah kenapa saya bilang begitu." (complete thought)

=== CLIP DURATION ===
- Ideal: 30–90 seconds
- Minimum: 15 seconds  
- Maximum: 120 seconds
- If a topic needs 100+ seconds, capture it ALL — do NOT cut short

=== WHAT MAKES A VIRAL CLIP ===
- Shocking revelations or confessions
- Controversial opinions that spark debate
- Emotional moments (anger, tears, laughter)
- Expert knowledge — surprising facts
- Funny moments, roasts, comebacks
- Relatable struggles or universal truths
- Complete story arcs with a beginning, middle, and end

=== TIMESTAMP GUIDE ===
[02:30] = 150.0 seconds, [10:45] = 645.0 seconds
Use the nearest [MM:SS] marker to estimate times. Output in SECONDS.

=== OUTPUT FORMAT ===
Respond with ONLY a raw JSON array. No markdown, no backticks, no explanation.
Each object:
{{
    "start_time": float,
    "end_time": float,
    "virality_score": int (0-100),
    "description": "Why this segment will go viral",
    "title": "Clickbait title for the clip",
    "hashtags": ["#relevant", "#tags"],
    "layout": "portrait_face"
}}

=== TRANSCRIPT ===
{transcript_text}"""
