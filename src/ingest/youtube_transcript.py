"""
ViralClip AI - YouTube Transcript Fetcher
Handles prioritizing direct transcript extraction without downloading video.
"""
from typing import List, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
import json

from src.models import TranscriptSegment

def get_youtube_transcript(video_id: str, languages: List[str] = ['id', 'en']) -> Optional[List[TranscriptSegment]]:
    """
    Fetches the transcript directly from YouTube's API if available.
    Returns a list of TranscriptSegments or None if not available.
    """
    try:
        # Fetch transcript directly using the new v1.2.4 API syntax
        api = YouTubeTranscriptApi()
        raw_transcript = api.fetch(video_id, languages=languages)
        
        # Convert to our internal structure
        segments = []
        for segment in raw_transcript:
            # Handle both dictionary (old API) and FetchedTranscriptSnippet (new API)
            text = segment['text'] if isinstance(segment, dict) else segment.text
            start = segment['start'] if isinstance(segment, dict) else segment.start
            duration = segment['duration'] if isinstance(segment, dict) else segment.duration
            
            segments.append(TranscriptSegment(
                start=start,
                end=start + duration,
                text=text
            ))
            
        return segments
        
    except Exception as e:
        print(f"[-] Could not fetch YouTube transcript for {video_id}: {e}")
        return None

def extract_video_id(url: str) -> str:
    """Helper to extract video ID from various YouTube URL formats."""
    import re
    # Match patterns like v= or youtu.be/
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return ""
