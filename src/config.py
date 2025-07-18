"""
Centralized configuration management for the project.
Handles file paths and environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Dynamically find the project root by going up one level from this file's parent (src)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_ROOT / ".env", override=True)
print(f'[DEBUG] Attempting to load .env from: {PROJECT_ROOT / ".env"}')

# --- Path Configurations (relative to project root) ---
AUDIO_DIR = PROJECT_ROOT / "audio"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
DIARIZED_TRANSCRIPTS_DIR = PROJECT_ROOT / "diarized_transcripts"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
SUMMARIES_DIR = PROJECT_ROOT / "summaries"
PDF_OUTPUT_DIR = PROJECT_ROOT / "pdf_outputs"
DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "metadata.db"

# --- Model & API Configurations from .env ---
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH")
LLM_GGUF_REPO = os.getenv("LLM_GGUF_REPO")
LLM_GGUF_FILENAME = os.getenv("LLM_GGUF_FILENAME")
COMPUTE_DEVICE_SETTING = os.getenv("COMPUTE_DEVICE", "auto").lower()
WHISPER_INITIAL_PROMPT = os.getenv("WHISPER_INITIAL_PROMPT", "") # Prompt for Whisper

print(f"[DEBUG] Config LLM_GGUF_REPO: {LLM_GGUF_REPO}")
print(f"[DEBUG] Config LLM_GGUF_FILENAME: {LLM_GGUF_FILENAME}")
print(f"[DEBUG] Config COMPUTE_DEVICE_SETTING: {COMPUTE_DEVICE_SETTING}")

_TORCH_DEVICE = None

def get_compute_device():
    """
    Determines the optimal compute device based on system capabilities and .env settings.
    Caches the result for subsequent calls.
    Priority: CUDA > MPS > CPU.
    The user can override this by setting COMPUTE_DEVICE in .env.
    Returns a torch.device object.
    """
    global _TORCH_DEVICE
    if _TORCH_DEVICE is not None:
        return _TORCH_DEVICE

    import torch
    
    selected_device_str = "cpu" # Default to CPU

    if COMPUTE_DEVICE_SETTING == "cuda" and torch.cuda.is_available():
        print("Device forced to CUDA by .env setting.")
        selected_device_str = "cuda"
    elif COMPUTE_DEVICE_SETTING == "mps" and torch.backends.mps.is_available():
        print("Device forced to MPS by .env setting.")
        selected_device_str = "mps"
    elif COMPUTE_DEVICE_SETTING == "cpu":
        print("Device forced to CPU by .env setting.")
        selected_device_str = "cpu"
    elif COMPUTE_DEVICE_SETTING == "auto":
        if torch.cuda.is_available():
            print("Auto-detected CUDA device.")
            selected_device_str = "cuda"
        elif torch.backends.mps.is_available():
            print("Auto-detected Apple MPS device.")
            selected_device_str = "mps"
        else:
            print("Auto-detected CPU.")
            selected_device_str = "cpu"
    else:
        print(f"Warning: COMPUTE_DEVICE setting '{COMPUTE_DEVICE_SETTING}' is invalid. Falling back to CPU.")
        selected_device_str = "cpu"

    _TORCH_DEVICE = torch.device(selected_device_str)
    return _TORCH_DEVICE


def ensure_dirs_exist():
    """Ensures that all output directories exist."""
    for path in [TRANSCRIPTS_DIR, DIARIZED_TRANSCRIPTS_DIR, EMBEDDINGS_DIR, SUMMARIES_DIR, PDF_OUTPUT_DIR, DATABASE_DIR]:
        path.mkdir(parents=True, exist_ok=True)