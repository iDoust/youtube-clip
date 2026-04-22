<p align="center">
  <h1 align="center">🎬 ViralClip AI</h1>
  <p align="center">
    <strong>Turn long YouTube videos into viral short-form clips — fully automated.</strong>
  </p>
  <p align="center">
    Paste a YouTube URL → AI finds the most viral moments → Downloads only those segments → Smart crops to 9:16 → Burns TikTok-style subtitles → Exports ready-to-post clips.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/AI-LLM%20Powered-blueviolet?logo=openai&logoColor=white" alt="LLM Powered">
  <img src="https://img.shields.io/badge/Whisper-faster--whisper-orange?logo=openai&logoColor=white" alt="Faster Whisper">
  <img src="https://img.shields.io/badge/FFmpeg-8.0-green?logo=ffmpeg&logoColor=white" alt="FFmpeg">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="MIT License">
</p>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **AI Viral Detection** | LLM analyzes the full transcript to find the most share-worthy moments |
| ⚡ **Zero Full Downloads** | Stage A only fetches the transcript — no video download needed for analysis |
| 🎯 **Segment-Only Download** | Downloads only the 30-90s clips, not the full 80-minute podcast |
| 📐 **Smart Crop (16:9 → 9:16)** | Face-aware cropping using YuNet + PySceneDetect for seamless scene transitions |
| 💬 **TikTok-Style Subtitles** | Word-by-word karaoke captions via Whisper with sentence-boundary trimming |
| 🌐 **Subtitle Translation** | Translate subtitles to any language (e.g., Indonesian) via LLM |
| 🏷️ **Watermark & Branding** | Add text or image watermarks to final clips |
| 🔄 **LLM Fallback Chain** | OpenAI-compatible API (primary) → Gemini (fallback) — never fails silently |
| 🍪 **Anti-Bot Bypass** | Automatic browser cookie retry + FFmpeg direct stream fallback for protected videos |

---

## 🏗️ Architecture

```
youtube-clip/
├── main.py                          # CLI entry point — orchestrates the full pipeline
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
│
├── src/
│   ├── config.py                    # Global config (paths, API keys from .env)
│   ├── models.py                    # Pydantic data models (ViralClip, TranscriptSegment)
│   │
│   ├── ingest/                      # STAGE A: Content Ingestion
│   │   ├── youtube_transcript.py    # Fetch transcript via YouTube Transcript API
│   │   ├── transcriber.py           # Whisper fallback transcription (faster-whisper)
│   │   └── downloader.py            # yt-dlp segment/audio/full download + fallbacks
│   │
│   ├── detection/                   # STAGE B: Viral Moment Detection
│   │   └── viral_analyzer.py        # LLM orchestrator with boundary post-validation
│   │
│   ├── llm_providers/               # Pluggable LLM backends
│   │   ├── __init__.py              # Abstract base class + shared prompt builder
│   │   ├── openrouter_provider.py   # OpenAI-compatible API (OpenRouter, MiniMax, etc.)
│   │   └── gemini_provider.py       # Google Gemini via google-genai SDK
│   │
│   ├── visual/                      # STAGE C: Visual Processing
│   │   └── smart_crop.py            # PySceneDetect + YuNet face-based 9:16 cropping
│   │
│   └── editing/                     # STAGE D: Post-Production
│       ├── whisper_subtitles.py     # Whisper word-level subtitles + boundary detection
│       ├── caption_renderer.py      # ASS/SRT karaoke subtitle generator + FFmpeg burner
│       ├── translator.py            # LLM-powered batch subtitle translation
│       └── branding.py              # Text/image watermark overlay
│
├── bin/                             # Bundled FFmpeg binary (not in git)
├── models/                          # ML models (YuNet face detection ONNX)
├── temp/                            # Intermediate processing files (auto-cleaned)
└── outputs/                         # Final exported clips
```

---

## 🔄 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STAGE A: INGEST                              │
│                                                                     │
│   YouTube URL ──► YouTube Transcript API (instant, no download)     │
│                          │                                          │
│                     [if unavailable]                                │
│                          │                                          │
│                   yt-dlp audio-only ──► faster-whisper (large-v3)   │
│                          │                                          │
│                   FullTranscript (segments with timestamps)         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                     STAGE B: VIRAL DETECTION                        │
│                                                                     │
│   FullTranscript ──► LLM Analysis (OpenRouter → Gemini fallback)    │
│                          │                                          │
│              Prompt: skip first 2 min, find complete stories,       │
│              score virality (0-100), output JSON                    │
│                          │                                          │
│              Post-validation: fix mid-sentence boundaries           │
│                          │                                          │
│                   List[ViralClip] (sorted by score)                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                    [if --render flag]
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                   STAGE C+D: RENDER (per clip)                      │
│                                                                     │
│   C1: yt-dlp segment download (only 30-90s, +10s buffer)            │
│       └► cookie retry → FFmpeg direct stream fallback               │
│                                                                     │
│   C2: Smart Crop (16:9 → 9:16)                                      │
│       ├► PySceneDetect → find camera cuts                           │
│       ├► YuNet face detection (per scene, multi-sample vote)        │
│       ├► Scenario A: 1 face → face-anchored center crop             │
│       └► Scenario B: 2 faces → split & merge sides                  │
│                                                                     │
│   D1: Whisper subtitle pipeline                                     │
│       ├► Transcribe buffered clip (word-level timestamps)           │
│       ├► Find best sentence-ending boundary                         │
│       ├► Trim video to boundary                                     │
│       ├► Generate ASS karaoke subtitles (2-3 words per group)       │
│       └► Burn subtitles with FFmpeg                                 │
│                                                                     │
│   D2: Watermark (optional, --watermark flag)                        │
│                                                                     │
│   D3: Export to outputs/ directory                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **FFmpeg 8.0+** — place `ffmpeg.exe` in `bin/` or install system-wide
- **Node.js** — required by yt-dlp for YouTube SABR decryption
- At least one LLM API key (see [Configuration](#-configuration))

### Installation

```bash
# Clone the repository
git clone https://github.com/iDoust/youtube-clip.git
cd youtube-clip

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy the environment template and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Primary LLM — any OpenAI-compatible API (OpenRouter, MiniMax, etc.)
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your_api_key_here
LLM_MODEL=meta-llama/llama-3.3-70b-instruct

# Gemini (Fallback LLM)
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-2.5-flash
```

> [!TIP]
> You only need **one** LLM provider configured. The system will use whatever is available.

### Download YuNet Model

Place the face detection model in the `models/` directory:

```
models/face_detection_yunet_2023mar.onnx
```

Download from: [OpenCV Zoo — YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)

---

## 📖 Usage

### Analysis Only (No Download)

Find viral moments without downloading any video:

```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --clips 5
```

Output: A table showing the top 5 viral moments with scores, timestamps, titles, and explanations.

### Full Pipeline (Analysis + Render)

Download segments, crop, subtitle, and export:

```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --clips 3 --render
```

### With Subtitle Translation

Translate subtitles to Indonesian (or any language):

```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --clips 3 --render --translate id
```

### With Watermark

Add branding text to the final clips:

```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --clips 3 --render --watermark "MyBrand"
```

### Full Example

```bash
python main.py \
  --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  --clips 5 \
  --render \
  --translate id \
  --watermark "ViralClip AI"
```

---

## ⚙️ CLI Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--url` | `str` | *required* | YouTube video URL |
| `--clips` | `int` | `3` | Number of viral clips to find |
| `--render` | `flag` | `false` | Enable download + render pipeline (Stage C+D) |
| `--translate` | `str` | `None` | Translate subtitles to language code (e.g., `id`, `en`, `es`) |
| `--watermark` | `str` | `None` | Add text watermark to final clips |

---

## 🧠 How the AI Works

### Viral Detection Prompt

The LLM receives the full transcript as flowing text with timestamp markers every ~30 seconds. It is instructed to:

1. **Skip the first 2 minutes** — avoids selecting the video's own hook/intro
2. **Identify complete topic arcs** — never cut mid-sentence or mid-story
3. **Score virality (0-100)** based on: shocking revelations, controversial opinions, emotional moments, expert knowledge, humor, and relatability
4. **Output structured JSON** with timestamps, scores, titles, hashtags, and layout hints

### Boundary Post-Validation

After the LLM returns clips, `ViralAnalyzer` programmatically fixes boundaries:
- If `end_time` cuts mid-sentence → extends to the next sentence end (max +15s)
- If `start_time` cuts mid-sentence → pulls back to the previous sentence start (max -10s)

### Smart Crop Scenarios

| Scenario | Condition | Behavior |
|---|---|---|
| **A** (1 person) | ≤1 face detected (majority vote across 3 samples) | Face-anchored center crop to 9:16 |
| **B** (2 persons) | ≥2 faces detected, separated by >15% frame width | Split left/right panels, discard center gap, merge to 9:16 |
| **Fallback** | No faces or detection failure | Dead center crop |

---

## 🔌 LLM Providers

ViralClip AI supports a pluggable LLM backend with automatic failover:

| Provider | Config Keys | Notes |
|---|---|---|
| **OpenAI-compatible** (primary) | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Works with OpenRouter, MiniMax, local Ollama, etc. |
| **Google Gemini** (fallback) | `GEMINI_API_KEY`, `GEMINI_MODEL` | Uses `google-genai` SDK, great for large contexts |

The system tries providers in order and falls back automatically on failure.

---

## 📦 Dependencies

| Category | Package | Purpose |
|---|---|---|
| **Ingest** | `youtube-transcript-api` | Fetch YouTube transcripts (no download) |
| | `yt-dlp` | Download video/audio segments from YouTube |
| | `faster-whisper` | CTranslate2-based Whisper for transcription |
| **AI / LLM** | `openai` | OpenAI-compatible API client (OpenRouter, MiniMax) |
| | `google-genai` | Google Gemini API client |
| **Vision** | `opencv-python` | YuNet face detection, video frame analysis |
| | `scenedetect` | PySceneDetect — camera cut detection |
| **Data** | `pydantic` | Data models & validation |
| | `python-dotenv` | `.env` file loading |
| **CLI** | `rich` | Terminal UI (progress bars, tables, colored output) |
| **External** | FFmpeg 8.0+ | Video processing, cropping, subtitle burning |

---

## 📂 Output

Final clips are saved to the `outputs/` directory:

```
outputs/
├── VIDEO_ID_clip1_final.mp4
├── VIDEO_ID_clip2_final.mp4
└── VIDEO_ID_clip3_final.mp4
```

Each clip is:
- **9:16 portrait** format (optimized for TikTok, Reels, Shorts)
- **Sentence-boundary trimmed** (no mid-sentence cutoffs)
- **Subtitled** with word-by-word karaoke captions
- **Watermarked** (if `--watermark` was used)

---

## 🛣️ Roadmap

- [ ] GPU acceleration for Whisper and FFmpeg
- [ ] Batch processing (multiple URLs)
- [ ] Web UI dashboard
- [ ] Auto-upload to TikTok, YouTube Shorts, Instagram Reels
- [ ] Speaker diarization for multi-speaker videos
- [ ] Background music overlay
- [ ] Custom subtitle styles (fonts, colors, animations)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

