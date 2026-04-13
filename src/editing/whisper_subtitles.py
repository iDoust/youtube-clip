"""
Whisper Subtitle Generator
Transcribes a video clip's audio using faster-whisper for accurate word-level subtitles.
Replaces YouTube transcript (which has overlapping/garbled segments) with clean Whisper output.
Also provides sentence boundary detection for clip trimming.

Supports: Indonesian (id), English (en), auto-detect
"""
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from rich.console import Console

console = Console()


def transcribe_clip_for_subtitles(
    video_path: str,
    language: str = None,
    model_size: str = "small"
) -> List[Dict]:
    """
    Transcribe a video clip's audio using faster-whisper.
    Returns list of subtitle segments with word-level timing.
    
    Args:
        video_path: Path to the video clip (mp4)
        language: Language code ('id', 'en', or None for auto-detect)
        model_size: Whisper model size ('small' = fast, 'medium' = balanced, 'large-v3' = best)
    
    Returns:
        List of dicts with 'start', 'end', 'text' keys
        Each segment is a natural sentence/phrase from Whisper
    """
    from faster_whisper import WhisperModel
    
    video_path = str(video_path)
    
    # Extract audio from video to temp WAV
    audio_path = Path(video_path).with_suffix('.wav')
    _extract_audio(video_path, str(audio_path))
    
    if not audio_path.exists():
        console.print("[red][-] Failed to extract audio from clip[/red]")
        return []
    
    try:
        console.print(f"[cyan][*] Loading Whisper model: {model_size}...[/cyan]")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        # Transcribe with word-level timestamps
        transcribe_opts = {
            "word_timestamps": True,
            "vad_filter": True,  # Voice Activity Detection — skips silence
        }
        if language:
            transcribe_opts["language"] = language
        
        console.print(f"[cyan][*] Transcribing clip audio ({language or 'auto-detect'})...[/cyan]")
        segments_iter, info = model.transcribe(str(audio_path), **transcribe_opts)
        
        detected_lang = info.language
        console.print(f"[cyan]    Detected language: {detected_lang} (prob={info.language_probability:.2f})[/cyan]")
        
        # Collect all segments with word-level data
        subtitle_segments = []
        for segment in segments_iter:
            # Use word-level timestamps for precise subtitle timing
            if segment.words:
                # Group words into natural subtitle chunks (2-4 words)
                words = list(segment.words)
                subtitle_segments.extend(
                    _group_words_to_subtitles(words)
                )
            else:
                # Fallback: use segment-level timing
                text = segment.text.strip()
                if text and len(text) >= 2:
                    subtitle_segments.append({
                        "start": round(segment.start, 3),
                        "end": round(segment.end, 3),
                        "text": text
                    })
        
        console.print(f"[green][+] Whisper transcription: {len(subtitle_segments)} subtitle segments[/green]")
        return subtitle_segments
        
    except Exception as e:
        console.print(f"[red][-] Whisper transcription failed: {e}[/red]")
        return []
    finally:
        # Cleanup temp audio
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)


def find_best_end_boundary(
    video_path: str,
    original_duration: float,
    buffer_duration: float = 10.0,
    language: str = None,
    model_size: str = "small"
) -> Tuple[float, List[Dict]]:
    """
    Transcribe a clip (which was downloaded with buffer) and find
    the best sentence-ending point near `original_duration`.
    
    Returns:
        (best_end_time, subtitle_segments)
        - best_end_time: the time of the last complete sentence near original_duration
        - subtitle_segments: Whisper word-grouped subtitles for the final clip
    """
    from faster_whisper import WhisperModel
    
    audio_path = Path(video_path).with_suffix('.wav')
    _extract_audio(str(video_path), str(audio_path))
    
    if not audio_path.exists():
        return original_duration, []
    
    try:
        console.print(f"[cyan][*] Loading Whisper model: {model_size}...[/cyan]")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        transcribe_opts = {"word_timestamps": True, "vad_filter": True}
        if language:
            transcribe_opts["language"] = language
        
        console.print(f"[cyan][*] Transcribing clip audio ({language or 'auto-detect'})...[/cyan]")
        segments_iter, info = model.transcribe(str(audio_path), **transcribe_opts)
        
        detected_lang = info.language
        console.print(f"[cyan]    Detected language: {detected_lang} (prob={info.language_probability:.2f})[/cyan]")
        
        # Collect ALL segments with sentence-level info
        all_sentences = []  # [{start, end, text}] — full sentences from Whisper
        all_words = []      # for subtitle generation
        
        for segment in segments_iter:
            text = segment.text.strip()
            if text and len(text) >= 2:
                all_sentences.append({
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": text
                })
            if segment.words:
                all_words.extend(list(segment.words))
        
        # Find the best end boundary near original_duration
        # Strategy: find the last sentence that ENDS before or near original_duration
        # Then check if the NEXT sentence also ends close — if so, include it
        best_end = original_duration
        
        for i, sent in enumerate(all_sentences):
            if sent['end'] <= original_duration + 3.0:
                # This sentence ends within the original clip range (+3s tolerance)
                best_end = sent['end']
            elif sent['start'] <= original_duration:
                # Sentence starts within range but extends past it
                # Include it — don't cut mid-sentence!
                best_end = sent['end']
            else:
                # This sentence starts after original_duration — stop
                break
        
        # Generate subtitle word-groups only up to best_end
        trimmed_words = [w for w in all_words if w.end <= best_end + 0.5]
        subtitle_segments = _group_words_to_subtitles(trimmed_words) if trimmed_words else []
        
        # Also trim subtitle_segments to best_end
        subtitle_segments = [s for s in subtitle_segments if s['start'] < best_end]
        for s in subtitle_segments:
            s['end'] = min(s['end'], best_end)
        
        if abs(best_end - original_duration) > 1.0:
            delta = best_end - original_duration
            sign = '+' if delta > 0 else ''
            console.print(f"[cyan]    Boundary adjusted: {sign}{delta:.1f}s (sentence-aligned)[/cyan]")
        
        console.print(f"[green][+] Whisper: {len(subtitle_segments)} subtitle groups, end={best_end:.1f}s[/green]")
        return best_end, subtitle_segments
        
    except Exception as e:
        console.print(f"[red][-] Whisper boundary detection failed: {e}[/red]")
        return original_duration, []
    finally:
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)


def _extract_audio(video_path: str, audio_path: str):
    """Extract audio track from video to WAV using FFmpeg."""
    from src.config import config
    
    ffmpeg_bin = config.BASE_DIR / "bin" / "ffmpeg.exe"
    ffmpeg_cmd = str(ffmpeg_bin) if ffmpeg_bin.exists() else "ffmpeg"
    
    cmd = [
        ffmpeg_cmd, '-y',
        '-i', video_path,
        '-vn',                    # No video
        '-acodec', 'pcm_s16le',   # WAV format
        '-ar', '16000',           # 16kHz (Whisper optimal)
        '-ac', '1',               # Mono
        audio_path
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def _group_words_to_subtitles(words, max_words=3) -> List[Dict]:
    """
    Group Whisper word-level data into subtitle chunks.
    Each chunk has 2-3 words, respecting punctuation as break points.
    
    This produces TikTok-style word-by-word subtitles that are
    accurate to the actual speech timing.
    """
    if not words:
        return []
    
    subtitles = []
    current_words = []
    chunk_start = None
    
    for word_info in words:
        word_text = word_info.word.strip()
        if not word_text:
            continue
        
        if chunk_start is None:
            chunk_start = word_info.start
        
        current_words.append(word_text)
        
        # Break on punctuation or max words
        is_sentence_end = word_text.endswith(('.', '!', '?'))
        is_pause = word_text.endswith(',')
        hit_max = len(current_words) >= max_words
        
        if is_sentence_end or is_pause or hit_max:
            text = " ".join(current_words)
            subtitles.append({
                "start": round(chunk_start, 3),
                "end": round(word_info.end, 3),
                "text": text
            })
            current_words = []
            chunk_start = None
    
    # Remaining words
    if current_words and chunk_start is not None:
        subtitles.append({
            "start": round(chunk_start, 3),
            "end": round(words[-1].end, 3),
            "text": " ".join(current_words)
        })
    
    return subtitles
