"""
ViralClip AI - Downloader
Handles downloading audio (for Whisper fallback) and video segments (for rendering clips).
Uses browser cookies to bypass YouTube anti-bot protection for long videos (1hr+ podcasts).
"""
import yt_dlp
import subprocess
from pathlib import Path
from typing import Optional, List
from src.config import config


def _get_base_opts() -> dict:
    """
    Returns base yt-dlp options.
    Cookies are NOT included by default — they are added as a retry strategy
    only if the initial download fails with a 403 error.
    """
    opts = {
        'quiet': False,
        'no_warnings': True,
        'noprogress': False,
        # Use Node.js to decrypt YouTube's SABR-protected streams
        # Without this, many videos return 403 Forbidden
        'extractor_args': {'youtube': {'js_runtimes': ['nodejs']}},
    }
    
    # Set FFmpeg location if bundled in bin/
    ffmpeg_path = config.BASE_DIR / "bin" / "ffmpeg.exe"
    if ffmpeg_path.exists():
        opts['ffmpeg_location'] = str(config.BASE_DIR / "bin")
    
    return opts

def _get_cookie_opts() -> dict:
    """
    Returns yt-dlp options WITH browser cookies for authentication.
    Used as a retry when downloads fail with 403 Forbidden.
    """
    for browser in ['chrome', 'edge', 'firefox', 'brave', 'opera']:
        try:
            # Test if we can access this browser's cookies
            test_opts = {**_get_base_opts(), 'cookiesfrombrowser': (browser,), 'quiet': True, 'simulate': True}
            with yt_dlp.YoutubeDL(test_opts) as ydl:
                pass
            print(f"[+] Using cookies from {browser} browser for authentication.")
            return {'cookiesfrombrowser': (browser,)}
        except Exception:
            continue
    
    print("[-] No browser cookies available. Download may fail for protected videos.")
    return {}


def download_audio_only(url: str, output_filename: str = "temp_audio") -> Optional[Path]:
    """
    Downloads only the best audio stream and converts to WAV.
    Used as fallback when YouTube transcript API is not available.
    Returns the Path to the downloaded audio file, or None if failed.
    """
    output_path = config.TEMP_DIR / f"{output_filename}.wav"
    
    ydl_opts = {
        **_get_base_opts(),
        'format': 'bestaudio/best',
        'outtmpl': str(config.TEMP_DIR / f"{output_filename}.%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    try:
        print(f"[*] Downloading audio for {url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        if output_path.exists():
            return output_path
        else:
            print("[-] Audio file not found after download.")
            return None
            
    except Exception as e:
        print(f"[-] Download failed: {e}")
        return None


def download_clip_segment(url: str, start_time: float, end_time: float, 
                          output_filename: str = "clip") -> Optional[Path]:
    """
    Downloads ONLY a specific time segment of a video (e.g., 30 seconds out of 80 minutes).
    This is the key function that makes long podcast videos efficient to process.
    
    For an 80-minute podcast where AI found a viral moment at 45:00-45:30,
    this function downloads ONLY those 30 seconds — not the full 80 minutes.
    
    Returns the Path to the downloaded clip, or None if failed.
    """
    output_path = config.TEMP_DIR / f"{output_filename}.mp4"
    
    # Format time range for yt-dlp's --download-sections
    time_range = f"*{_format_time(start_time)}-{_format_time(end_time)}"
    
    ydl_opts = {
        **_get_base_opts(),
        'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': str(output_path),
        'download_ranges': yt_dlp.utils.download_range_func(None, [{
            'start_time': start_time,
            'end_time': end_time,
        }]),
        'force_keyframes_at_cuts': True,
        'merge_output_format': 'mp4',
    }

    try:
        print(f"[*] Downloading segment {_format_time(start_time)} - {_format_time(end_time)}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[+] Clip segment saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return output_path
        else:
            raise Exception("Downloaded file is empty or missing")
            
    except Exception as e:
        print(f"[-] Segment download failed: {e}")
        
        # Retry with browser cookies (fixes 403 Forbidden on protected videos)
        print("[*] Retrying with browser cookies...")
        cookie_opts = _get_cookie_opts()
        if cookie_opts:
            try:
                ydl_opts.update(cookie_opts)
                # Clean up failed partial file
                if output_path.exists():
                    output_path.unlink()
                    
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                    
                if output_path.exists() and output_path.stat().st_size > 0:
                    print(f"[+] Clip segment saved with cookies: {output_path}")
                    return output_path
            except Exception as e2:
                print(f"[-] Cookie retry also failed: {e2}")
        
        # Final fallback: FFmpeg direct stream
        print("[*] Trying FFmpeg direct stream fallback...")
        return _download_clip_ffmpeg_fallback(url, start_time, end_time, output_path)


def _download_clip_ffmpeg_fallback(url: str, start_time: float, end_time: float, 
                                    output_path: Path) -> Optional[Path]:
    """
    Fallback: Use yt-dlp to get direct stream URLs for video-only and audio-only,
    then use FFmpeg to download only the specific segment and merge them.
    YouTube no longer provides combined video+audio via direct URL.
    """
    try:
        # Step 1: Get format info from yt-dlp
        info_opts = {
            **_get_base_opts(),
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        
        formats = info.get('formats', [])
        
        # Find best video-only stream with direct URL (not m3u8)
        video_url = None
        for f in sorted(formats, key=lambda x: x.get('height', 0) or 0, reverse=True):
            fmt_url = f.get('url', '')
            if (fmt_url and '.m3u8' not in fmt_url and 
                f.get('vcodec', 'none') != 'none' and 
                f.get('ext') in ('mp4', 'webm')):
                video_url = fmt_url
                print(f"    [+] Found video stream: {f.get('resolution', '?')} {f.get('vcodec', '?')}")
                break
        
        # Find best audio-only stream with direct URL
        audio_url = None
        for f in sorted(formats, key=lambda x: x.get('abr', 0) or 0, reverse=True):
            fmt_url = f.get('url', '')
            if (fmt_url and '.m3u8' not in fmt_url and
                f.get('acodec', 'none') != 'none' and 
                f.get('vcodec') in ('none', None) and
                f.get('ext') in ('m4a', 'webm', 'mp4')):
                audio_url = fmt_url
                print(f"    [+] Found audio stream: {f.get('abr', '?')}kbps {f.get('acodec', '?')}")
                break
        
        if not video_url:
            print("[-] Could not find any video stream with a direct URL.")
            return None
            
        # Step 2: Use FFmpeg to download segment(s) and merge
        duration = end_time - start_time
        ffmpeg_bin = config.BASE_DIR / "bin" / "ffmpeg.exe"
        ffmpeg_cmd = str(ffmpeg_bin) if ffmpeg_bin.exists() else "ffmpeg"
        
        if audio_url:
            # Download video + audio separately and merge
            cmd = [
                ffmpeg_cmd, '-y',
                '-ss', str(start_time), '-i', video_url,
                '-ss', str(start_time), '-i', audio_url,
                '-t', str(duration),
                '-map', '0:v:0', '-map', '1:a:0',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                '-movflags', '+faststart',
                str(output_path)
            ]
        else:
            # Video only (no audio found)
            cmd = [
                ffmpeg_cmd, '-y',
                '-ss', str(start_time), '-i', video_url,
                '-t', str(duration),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-movflags', '+faststart',
                str(output_path)
            ]
        
        print(f"[*] FFmpeg fallback: downloading {duration:.0f}s segment...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if output_path.exists() and output_path.stat().st_size > 0:
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"[+] FFmpeg fallback succeeded: {output_path} ({size_mb:.1f} MB)")
            return output_path
        else:
            err_msg = result.stderr[-300:] if result.stderr else 'unknown error'
            print(f"[-] FFmpeg fallback failed: {err_msg}")
            return None
            
    except Exception as e:
        print(f"[-] FFmpeg fallback failed: {e}")
        return None


def download_full_video(url: str, output_filename: str = "full_video", 
                        max_quality: int = 720) -> Optional[Path]:
    """
    Downloads the full video file. Only used when segment download is not possible
    or when the user explicitly wants the full video.
    Quality is capped to save bandwidth (default: 720p).
    """
    output_path = config.TEMP_DIR / f"{output_filename}.mp4"
    
    ydl_opts = {
        **_get_base_opts(),
        'format': f'bestvideo[height<={max_quality}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_quality}][ext=mp4]/best',
        'outtmpl': str(output_path),
        'merge_output_format': 'mp4',
    }

    try:
        print(f"[*] Downloading full video ({max_quality}p max)...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        if output_path.exists():
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"[+] Full video saved: {output_path} ({size_mb:.1f} MB)")
            return output_path
        else:
            print("[-] Video file not found after download.")
            return None
            
    except Exception as e:
        print(f"[-] Full video download failed: {e}")
        return None


def _format_time(seconds: float) -> str:
    """Formats seconds into HH:MM:SS.ms for yt-dlp time ranges."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
