"""
Branding Module
Handles adding visual watermarks, logos, or text branding to the final video using FFmpeg.
"""
import subprocess
from pathlib import Path
from src.config import config


def _get_ffmpeg() -> str:
    """Returns the path to the bundled FFmpeg binary, or 'ffmpeg' if not found."""
    ffmpeg_bin = config.BASE_DIR / "bin" / "ffmpeg.exe"
    return str(ffmpeg_bin) if ffmpeg_bin.exists() else "ffmpeg"


class Bander:
    def __init__(self):
        pass

    def add_text_watermark(self, video_path: Path, output_path: Path, text: str = "ViralClip AI"):
        """
        Adds a text watermark to the top right of the video.
        """
        print(f"[*] Adding watermark '{text}' to {video_path.name}...")

        ffmpeg = _get_ffmpeg()

        drawtext = (
            f"drawtext=text='{text}':fontcolor=white:fontsize=48:"
            f"box=1:boxcolor=black@0.5:boxborderw=10:"
            f"x=w-tw-20:y=20"
        )

        cmd = [
            ffmpeg, '-y',
            '-i', str(video_path),
            '-vf', drawtext,
            '-c:a', 'copy',
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"[-] Watermark failed: {result.stderr[-300:]}")

        return output_path

    def add_image_watermark(self, video_path: Path, logo_path: Path, output_path: Path):
        """
        Adds a static image logo (e.g., PNG) to the top right corner.
        """
        print(f"[*] Overlaying logo {logo_path.name} on {video_path.name}...")

        ffmpeg = _get_ffmpeg()

        cmd = [
            ffmpeg, '-y',
            '-i', str(video_path),
            '-i', str(logo_path),
            '-filter_complex', '[1:v]scale=100:-1[logo];[0:v][logo]overlay=W-w-20:20',
            '-c:a', 'copy',
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"[-] Image watermark failed: {result.stderr[-300:]}")

        return output_path
