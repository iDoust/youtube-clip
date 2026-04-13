"""
ViralClip AI - Main Entry Point
Full pipeline: URL → Transcript → Viral Detection → Download Segment → Smart Crop → Subtitle → Export
"""
import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.ingest.youtube_transcript import get_youtube_transcript, extract_video_id
from src.ingest.downloader import download_audio_only, download_clip_segment
from src.ingest.transcriber import transcribe_audio
from src.detection.viral_analyzer import ViralAnalyzer
from src.models import FullTranscript
from src.config import config

console = Console()

def main():
    parser = argparse.ArgumentParser(description="ViralClip AI - Turn long videos into viral clips")
    parser.add_argument("--url", required=True, help="YouTube Video URL")
    parser.add_argument("--clips", type=int, default=3, help="Number of viral clips to find (default: 3)")
    parser.add_argument("--render", action="store_true", help="Enable Stage C+D: download segments and render final clips")
    parser.add_argument("--translate", type=str, default=None, help="Translate subtitles to language code (e.g., 'id' for Indonesian)")
    parser.add_argument("--watermark", type=str, default=None, help="Add text watermark to output (e.g., 'ViralClip AI')")
    args = parser.parse_args()
    
    video_id = extract_video_id(args.url)
    if not video_id:
        console.print("[red]Invalid YouTube URL[/red]")
        sys.exit(1)
        
    console.print(f"\n[bold blue]{'='*50}[/bold blue]")
    console.print(f"[bold blue]  ViralClip AI Pipeline[/bold blue]")
    console.print(f"[bold blue]  Video: {video_id}[/bold blue]")
    console.print(f"[bold blue]{'='*50}[/bold blue]\n")
    
    # ---------------------------------------------------------
    # STAGE A: INGEST (Transcript only - NO video download)
    # ---------------------------------------------------------
    console.print("[bold yellow]--- STAGE A: INGEST ---[/bold yellow]")
    
    console.print("[cyan][*] Fetching YouTube transcript (no download needed)...[/cyan]")
    segments = get_youtube_transcript(video_id)
    source = "youtube_api"
    
    if segments:
        console.print(f"[green][+] Transcript fetched instantly! {len(segments)} segments retrieved.[/green]")
    else:
        # Fallback: Download audio and use Whisper
        console.print("[yellow][-] No YouTube transcript available. Falling back to Whisper...[/yellow]")
        console.print("[cyan][*] Downloading audio only (not full video)...[/cyan]")
        audio_path = download_audio_only(args.url, output_filename=video_id)
        
        if not audio_path:
            console.print("[red]Failed to download audio.[/red]")
            sys.exit(1)
            
        console.print("[cyan][*] Transcribing with faster-whisper...[/cyan]")
        segments = transcribe_audio(audio_path)
        source = "whisper"
        
        if not segments:
            console.print("[red]Failed to transcribe audio.[/red]")
            sys.exit(1)
            
    full_transcript = FullTranscript(video_id=video_id, source=source, segments=segments)
    console.print(f"[green][+] Ingest complete. {len(segments)} segments ready for analysis.[/green]\n")
    
    # ---------------------------------------------------------
    # STAGE B: VIRAL DETECTION (Text-only AI analysis)
    # ---------------------------------------------------------
    console.print("[bold yellow]--- STAGE B: VIRAL DETECTION ---[/bold yellow]")
    
    analyzer = ViralAnalyzer()
    try:
        console.print(f"[cyan][*] AI analyzing transcript for top {args.clips} viral moments...[/cyan]")
        viral_clips = analyzer.analyze(full_transcript, num_clips=args.clips)
        
        console.print(f"\n[bold green]Found {len(viral_clips)} Viral Clips:[/bold green]\n")
        
        table = Table(show_header=True, header_style="bold magenta", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", width=6)
        table.add_column("Time Range", width=18)
        table.add_column("Title", width=30)
        table.add_column("Why Viral", min_width=30)
        
        for i, clip in enumerate(viral_clips, 1):
            start_m = int(clip.start_time // 60)
            start_s = int(clip.start_time % 60)
            end_m = int(clip.end_time // 60)
            end_s = int(clip.end_time % 60)
            time_str = f"{start_m:02d}:{start_s:02d} → {end_m:02d}:{end_s:02d}"
            score_str = f"{'🔥' if clip.virality_score >= 80 else '⭐'} {clip.virality_score}"
            
            table.add_row(
                str(i),
                score_str,
                time_str,
                clip.title,
                clip.description[:80] + "..." if len(clip.description) > 80 else clip.description
            )
            
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Viral analysis failed:[/bold red] {e}")
        sys.exit(1)
    
    # ---------------------------------------------------------
    # STAGE C + D: RENDER (Only if --render flag is passed)
    # ---------------------------------------------------------
    if not args.render:
        console.print(f"\n[dim]Tip: Add --render to download segments and produce final clips.[/dim]")
        console.print(f"[dim]Example: python main.py --url \"{args.url}\" --clips {args.clips} --render[/dim]\n")
        return
    
    console.print(f"\n[bold yellow]--- STAGE C+D: RENDERING {len(viral_clips)} CLIPS ---[/bold yellow]")
    
    for i, clip in enumerate(viral_clips, 1):
        clip_name = f"{video_id}_clip{i}"
        duration = clip.end_time - clip.start_time
        
        console.print(f"\n[bold cyan]── Clip {i}/{len(viral_clips)}: \"{clip.title}\" ({duration:.0f}s) ──[/bold cyan]")
        
        # C1: Download segment with +10s buffer for Whisper boundary detection
        buffer_secs = 10.0
        console.print(f"[cyan][*] Downloading segment {clip.start_time:.0f}s-{clip.end_time:.0f}s (+{buffer_secs:.0f}s buffer)...[/cyan]")
        clip_path = download_clip_segment(
            args.url, 
            start_time=clip.start_time, 
            end_time=clip.end_time + buffer_secs,  # Download extra for boundary detection
            output_filename=clip_name
        )
        
        if not clip_path:
            console.print(f"[red][-] Failed to download segment for clip {i}. Skipping.[/red]")
            continue
        
        # C2: Smart Crop (16:9 → 9:16)
        try:
            from src.visual.smart_crop import SmartCrop
            
            console.print("[cyan][*] Applying Smart Crop (16:9 → 9:16)...[/cyan]")
            cropper = SmartCrop()
            cropped_path = config.TEMP_DIR / f"{clip_name}_cropped.mp4"
            
            result_path = cropper.process(str(clip_path), str(cropped_path))
            if result_path and Path(result_path).exists():
                clip_path = Path(result_path)
                console.print(f"[green][+] 9:16 smart crop applied successfully.[/green]")
            else:
                console.print("[yellow][-] Smart crop failed. Using original clip.[/yellow]")
                
        except Exception as e:
            console.print(f"[yellow][-] Crop skipped: {e}[/yellow]")
        
        # D1: Whisper transcription + sentence-boundary detection + subtitle burn
        try:
            from src.editing.caption_renderer import CaptionRenderer
            from src.editing.whisper_subtitles import find_best_end_boundary
            
            # Whisper transcribes the buffered clip, finds the best sentence-ending boundary,
            # and returns pre-trimmed subtitle segments
            console.print("[cyan][*] Whisper: transcribing + finding sentence boundary...[/cyan]")
            best_end, clip_segments = find_best_end_boundary(
                video_path=str(clip_path),
                original_duration=duration,  # Original clip duration (without buffer)
                buffer_duration=buffer_secs,
            )
            
            # Trim the video to the Whisper-determined boundary if different
            if best_end < (duration + buffer_secs - 1.0):
                try:
                    from src.config import config as cfg_trim
                    ffmpeg_bin = cfg_trim.BASE_DIR / "bin" / "ffmpeg.exe"
                    ffmpeg_cmd = str(ffmpeg_bin) if ffmpeg_bin.exists() else "ffmpeg"
                    
                    trimmed_path = config.TEMP_DIR / f"{clip_name}_trimmed.mp4"
                    import subprocess
                    subprocess.run([
                        ffmpeg_cmd, '-y',
                        '-i', str(clip_path),
                        '-t', str(best_end + 0.5),  # +0.5s for audio fade out
                        '-c', 'copy',
                        str(trimmed_path)
                    ], capture_output=True, text=True, timeout=30)
                    
                    if trimmed_path.exists() and trimmed_path.stat().st_size > 0:
                        clip_path = trimmed_path
                except Exception as e:
                    console.print(f"[yellow][-] Trim skipped: {e}[/yellow]")
            
            # Fallback: use simple Whisper if boundary detection failed
            if not clip_segments:
                console.print("[yellow][-] Boundary detection failed, falling back to simple Whisper...[/yellow]")
                from src.editing.whisper_subtitles import transcribe_clip_for_subtitles
                clip_segments = transcribe_clip_for_subtitles(
                    video_path=str(clip_path),
                    language=None,
                )
            
            if clip_segments:
                # Optional: translate subtitles
                if args.translate:
                    try:
                        from src.editing.translator import SubtitleTranslator
                        translator = SubtitleTranslator()
                        console.print(f"[cyan][*] Translating subtitles to '{args.translate}'...[/cyan]")
                        clip_segments = translator.translate_subtitles(clip_segments, target_lang=args.translate)
                    except Exception as e:
                        console.print(f"[yellow][-] Translation skipped: {e}[/yellow]")
                
                renderer = CaptionRenderer()
                
                # Get video dimensions for proper subtitle positioning
                try:
                    import cv2 as cv2_sub
                    cap_sub = cv2_sub.VideoCapture(str(clip_path))
                    vid_w = int(cap_sub.get(cv2_sub.CAP_PROP_FRAME_WIDTH))
                    vid_h = int(cap_sub.get(cv2_sub.CAP_PROP_FRAME_HEIGHT))
                    cap_sub.release()
                except Exception:
                    vid_w, vid_h = 1080, 1920  # Default 9:16
                
                ass_path = config.TEMP_DIR / f"{clip_name}.ass"
                renderer.generate_ass(clip_segments, ass_path, video_width=vid_w, video_height=vid_h)
                
                subtitled_path = config.TEMP_DIR / f"{clip_name}_subtitled.mp4"
                renderer.burn_subtitles(clip_path, ass_path, subtitled_path)
                
                if subtitled_path.exists() and subtitled_path.stat().st_size > 0:
                    clip_path = subtitled_path
                    console.print(f"[green][+] Whisper subtitles burned. ({len(clip_segments)} word-groups, end={best_end:.1f}s)[/green]")
            else:
                console.print("[yellow][-] No subtitle segments generated.[/yellow]")
                
        except Exception as e:
            console.print(f"[yellow][-] Subtitle rendering skipped: {e}[/yellow]")
        
        # D2: Add watermark/branding
        if args.watermark:
            try:
                from src.editing.branding import Bander
                brander = Bander()
                branded_path = config.TEMP_DIR / f"{clip_name}_branded.mp4"
                brander.add_text_watermark(clip_path, branded_path, text=args.watermark)
                
                if branded_path.exists():
                    clip_path = branded_path
                    console.print(f"[green][+] Watermark '{args.watermark}' added.[/green]")
            except Exception as e:
                console.print(f"[yellow][-] Watermark skipped: {e}[/yellow]")
        
        # D3: Move final clip to output directory
        import shutil
        final_path = config.OUTPUT_DIR / f"{clip_name}_final.mp4"
        shutil.copy2(clip_path, final_path)
        console.print(f"[bold green][✓] Final clip saved: {final_path}[/bold green]")
    
    # Summary
    console.print(f"\n[bold blue]{'='*50}[/bold blue]")
    console.print(f"[bold green]  Pipeline complete! {len(viral_clips)} clips rendered.[/bold green]")
    console.print(f"[bold blue]  Output: {config.OUTPUT_DIR}[/bold blue]")
    console.print(f"[bold blue]{'='*50}[/bold blue]\n")
    
    # Cleanup temp files
    console.print("[dim][*] Cleaning up temp files...[/dim]")
    for tmp_file in config.TEMP_DIR.iterdir():
        try:
            tmp_file.unlink()
        except Exception:
            pass

if __name__ == "__main__":
    main()
