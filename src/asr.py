
"""
ASR module using Whisper.
"""
import os
import whisper
import json
from datetime import datetime

def transcribe_audio(audio_path):
    """Transcribes an audio file using Whisper and saves the output as JSON."""
    from . import config
    from pathlib import Path

    device = config.get_compute_device()
    print(f"Using device: {device}")

    print(f"Loading Whisper model...")
    # Add logic to select device (CPU, CUDA, MPS)
    model = whisper.load_model("base", device=device) 
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path)

    # Save transcript
    from . import config
    config.ensure_dirs_exist()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = config.TRANSCRIPTS_DIR / f"{Path(audio_path).stem}_{timestamp}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    return output_path
