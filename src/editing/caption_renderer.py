"""
Caption Renderer Module
Creates TikTok/Reels-style word-by-word karaoke subtitles.
- Shows 2-3 words at a time in the BOTTOM area of the video
- Font scales to actual video resolution (no oversized text)
- De-overlaps segments to prevent stacking
"""
import subprocess
from pathlib import Path
from typing import List, Dict
from src.config import config

class CaptionRenderer:
    def __init__(self):
        pass

    def _deoverlap_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        YouTube transcript segments often have overlapping time ranges.
        This ensures each segment ends before the next one starts,
        preventing multiple subtitles from showing simultaneously.
        """
        if not segments:
            return segments
        
        # Sort by start time
        sorted_segs = sorted(segments, key=lambda s: s['start'])
        
        # Trim each segment's end to not exceed the next segment's start
        cleaned = []
        for i, seg in enumerate(sorted_segs):
            s = dict(seg)  # copy
            if i + 1 < len(sorted_segs):
                next_start = sorted_segs[i + 1]['start']
                if s['end'] > next_start:
                    s['end'] = next_start
            # Skip zero-duration or negative segments
            if s['end'] > s['start'] + 0.05:
                cleaned.append(s)
        
        return cleaned

    def _split_segment_to_words(self, segment: Dict) -> List[Dict]:
        """
        Splits a transcript segment into word-group entries.
        Respects punctuation as natural break points:
        - Period (.), question mark (?), exclamation (!): sentence boundary → end group
        - Comma (,): pause → end group (natural speech jeda)
        
        Shows 2-3 words per group maximum for readability.
        """
        words = segment['text'].strip().split()
        if not words:
            return []
        
        total_duration = segment['end'] - segment['start']
        if total_duration < 0.1:
            return []
        
        # For very short segments (< 1s) or few words, show all at once
        if total_duration < 1.0 or len(words) <= 2:
            return [segment]
        
        # Build chunks respecting punctuation boundaries
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            
            # Check if this word ends with punctuation (natural break point)
            is_sentence_end = word.rstrip().endswith(('.', '!', '?'))
            is_pause = word.rstrip().endswith(',')
            hit_max_words = len(current_chunk) >= 3
            
            if is_sentence_end or is_pause or hit_max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        # Don't forget remaining words
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        if not chunks:
            return [segment]
        
        # Distribute time proportionally to word count (more words = more time)
        total_words = sum(len(c.split()) for c in chunks)
        
        entries = []
        current_time = segment['start']
        
        for i, chunk in enumerate(chunks):
            # Time is proportional to number of words in this chunk
            word_count = len(chunk.split())
            chunk_duration = (word_count / total_words) * total_duration
            
            entry_end = current_time + chunk_duration
            # Last chunk ends exactly at segment end
            if i == len(chunks) - 1:
                entry_end = segment['end']
            
            entries.append({
                "start": round(current_time, 3),
                "end": round(entry_end, 3),
                "text": chunk
            })
            current_time = entry_end
        
        return entries

    @staticmethod
    def _clean_display_text(text: str) -> str:
        """Strip punctuation from display text. Punctuation is only used for grouping."""
        import re
        # Remove .  ,  !  ?  ;  : from the text
        cleaned = re.sub(r'[.,!?;:]', '', text)
        # Collapse multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def generate_ass(self, segments: List[Dict], output_path: Path, 
                     video_width: int = 1080, video_height: int = 1920):
        """
        Generates ASS subtitle file with proper TikTok-style formatting.
        Key design decisions:
        - Font size is ~5% of video height (readable but not intrusive)
        - Positioned at bottom 15% of screen (like the red box in TikTok)
        - Bold white text with black outline for readability over any background
        - Segments de-overlapped so only 1 subtitle shows at a time
        """
        # Step 1: De-overlap segments to prevent stacking
        clean_segments = self._deoverlap_segments(segments)
        
        # Step 2: Split into word-level groups
        word_entries = []
        for seg in clean_segments:
            word_entries.extend(self._split_segment_to_words(seg))
        
        # Step 3: De-overlap the word groups too (splitting can re-introduce overlaps)
        word_entries = self._deoverlap_segments(word_entries)
        
        def format_ass_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            centis = int((seconds - int(seconds)) * 100)
            return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"

        # Scale font size to video resolution
        # ~5% of video height = readable but not obnoxious
        font_size = max(12, int(video_height * 0.04))
        
        # Outline thickness scales too
        outline = max(1, int(font_size * 0.06))
        
        # Shadow distance
        shadow = 1
        
        # MarginV: position at bottom ~20% of screen (higher, in the red box area)
        margin_v = max(10, int(video_height * 0.20))
        
        # Side margins  
        margin_lr = max(10, int(video_width * 0.05))

        ass_content = f"""[Script Info]
Title: ViralClip Karaoke Subtitles
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial Black,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        for entry in word_entries:
            start = format_ass_time(entry['start'])
            end = format_ass_time(entry['end'])
            text = self._clean_display_text(entry['text']).upper()
            ass_content += f"Dialogue: 0,{start},{end},Karaoke,,0,0,0,,{text}\n"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
        
        print(f"[+] ASS karaoke file: {len(word_entries)} word groups, font={font_size}px, margin={margin_v}px")
        return output_path

    def generate_srt(self, segments: List[Dict], output_path: Path):
        """Legacy SRT generation with word-by-word splitting."""
        clean_segments = self._deoverlap_segments(segments)
        word_entries = []
        for seg in clean_segments:
            word_entries.extend(self._split_segment_to_words(seg))
        word_entries = self._deoverlap_segments(word_entries)
        
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        with open(output_path, "w", encoding="utf-8") as f:
            for i, entry in enumerate(word_entries, start=1):
                f.write(f"{i}\n")
                f.write(f"{format_time(entry['start'])} --> {format_time(entry['end'])}\n")
                f.write(f"{entry['text'].strip().upper()}\n\n")
                
        return output_path

    def burn_subtitles(self, video_path: Path, sub_path: Path, output_path: Path):
        """Burns ASS or SRT subtitles into the video using FFmpeg."""
        ffmpeg_bin = config.BASE_DIR / "bin" / "ffmpeg.exe"
        ffmpeg_cmd = str(ffmpeg_bin) if ffmpeg_bin.exists() else "ffmpeg"
        
        sub_escaped = str(sub_path).replace('\\', '/').replace(':', '\\:')
        ext = sub_path.suffix.lower()
        
        if ext == '.ass':
            vf = f"ass='{sub_escaped}'"
        else:
            style = (
                "FontName=Arial Black,FontSize=18,PrimaryColour=&H00FFFFFF,"
                "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Bold=-1,"
                "Alignment=2,MarginV=40"
            )
            vf = f"subtitles='{sub_escaped}':force_style='{style}'"
        
        cmd = [
            ffmpeg_cmd, '-y',
            '-i', str(video_path),
            '-vf', vf,
            '-c:a', 'copy',
            str(output_path)
        ]
        
        print(f"[*] Burning karaoke subtitles onto {video_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[+] Subtitles burned successfully.")
        else:
            err = result.stderr[-300:] if result.stderr else 'unknown error'
            print(f"[-] Subtitle burn failed: {err}")
        
        return output_path
