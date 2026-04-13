"""
ViralClip AI - Transcriber
Handles offline audio-to-text transcription using faster-whisper as a fallback.
"""
from faster_whisper import WhisperModel
from pathlib import Path
from typing import List, Optional

from src.models import TranscriptSegment

def transcribe_audio(audio_path: Path, language: str = "id") -> Optional[List[TranscriptSegment]]:
    """
    Transcribes audio using faster-whisper.
    Defaults to Indonesian ("id").
    """
    try:
        # We can configure this in config.py later, hardcoding for V1
        model_size = "large-v3"
        print(f"[*] Loading Whisper model: {model_size} (CPU mode)...")
        
        # device="cpu" avoids cuDNN DLL missing errors on Windows during testing
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        print(f"[*] Transcribing {audio_path.name}...")
        segments, info = model.transcribe(str(audio_path), language=language, word_timestamps=True)
        
        print(f"[*] Detected language: {info.language} with probability {info.language_probability:.2f}")
        
        transcript_segments = []
        for segment in segments:
            # We also capture word-level timestamps here for karaoke captions later
            words_list = []
            if segment.words:
                for word in segment.words:
                     words_list.append({
                         "start": word.start,
                         "end": word.end,
                         "word": word.word
                     })
                     
            transcript_segments.append(TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=words_list
            ))
            
        return transcript_segments
        
    except Exception as e:
        print(f"[-] Transcription failed: {e}")
        return None
