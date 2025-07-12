"""
Centralized configuration management for the project.
Handles file paths and environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Dynamically find the project root by going up one level from this file's parent (src)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")

# --- Path Configurations (relative to project root) ---
AUDIO_DIR = PROJECT_ROOT / "audio"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
DIARIZED_TRANSCRIPTS_DIR = PROJECT_ROOT / "diarized_transcripts"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
SUMMARIES_DIR = PROJECT_ROOT / "summaries"
DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "metadata.db"

# --- Model & API Configurations from .env ---
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH")
COMPUTE_DEVICE_SETTING = os.getenv("COMPUTE_DEVICE", "auto").lower()

_TORCH_DEVICE = None

def get_compute_device():
    """
    Determines the optimal compute device based on system capabilities and .env settings.
    Caches the result for subsequent calls.
    Priority: CUDA > MPS > CPU.
    The user can override this by setting COMPUTE_DEVICE in .env.
    """
    global _TORCH_DEVICE
    if _TORCH_DEVICE is not None:
        return _TORCH_DEVICE

    import torch
    
    if COMPUTE_DEVICE_SETTING == "cuda" and torch.cuda.is_available():
        print("Device forced to CUDA by .env setting.")
        _TORCH_DEVICE = "cuda"
    elif COMPUTE_DEVICE_SETTING == "mps" and torch.backends.mps.is_available():
        print("Device forced to MPS by .env setting.")
        _TORCH_DEVICE = "mps"
    elif COMPUTE_DEVICE_SETTING == "cpu":
        print("Device forced to CPU by .env setting.")
        _TORCH_DEVICE = "cpu"
    elif COMPUTE_DEVICE_SETTING == "auto":
        if torch.cuda.is_available():
            print("Auto-detected CUDA device.")
            _TORCH_DEVICE = "cuda"
        elif torch.backends.mps.is_available():
            print("Auto-detected Apple MPS device.")
            _TORCH_DEVICE = "mps"
        else:
            print("Auto-detected CPU.")
            _TORCH_DEVICE = "cpu"
    else:
        print(f"Warning: COMPUTE_DEVICE setting '{COMPUTE_DEVICE_SETTING}' is invalid. Falling back to CPU.")
        _TORCH_DEVICE = "cpu"

    return _TORCH_DEVICE


def ensure_dirs_exist():
    """Ensures that all output directories exist."""
    for path in [TRANSCRIPTS_DIR, DIARIZED_TRANSCRIPTS_DIR, EMBEDDINGS_DIR, SUMMARIES_DIR, DATABASE_DIR]:
        path.mkdir(parents=True, exist_ok=True)
