"""
ViralClip AI - Data Models
Defines all Pydantic schemas used across the application.
"""
from pydantic import BaseModel, Field
from typing import List, Optional

class ViralClip(BaseModel):
    start_time: float = Field(..., description="Start time of the clip in seconds")
    end_time: float = Field(..., description="End time of the clip in seconds")
    virality_score: int = Field(..., ge=0, le=100, description="Score defining how viral the clip is (0-100)")
    description: str = Field(..., description="Detailed explanation of why this clip is viral (Hook, Value, etc)")
    
    # Social Media Metadata
    title: str = Field(..., description="Clickbait, engaging title for the clip")
    hashtags: List[str] = Field(..., description="List of relevant hashtags")
    
    # Visual Layout
    layout: str = Field(default="portrait_face", description="Layout strategy (e.g., portrait_face, split_screen)")
    active_speaker_id: Optional[str] = Field(default=None, description="ID of the dominant speaker in this clip")

class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    words: Optional[List[dict]] = None # Expected to hold word-level timestamps if available

class FullTranscript(BaseModel):
    video_id: str
    source: str = Field(..., description="'youtube_api' or 'whisper'")
    segments: List[TranscriptSegment]
