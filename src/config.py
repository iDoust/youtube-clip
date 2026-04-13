"""
ViralClip AI - Configuration
Handles environment variables and global settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class Config:
    # Project Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    TEMP_DIR = BASE_DIR / "temp"
    OUTPUT_DIR = BASE_DIR / "outputs"

    # Ensure directories exist
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # API Keys
    # Primary LLM (OpenAI-compatible)
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")       # Fallback LLM
    HF_TOKEN = os.getenv("HF_TOKEN")                   # Reserved

    # Gemini model
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

config = Config()
