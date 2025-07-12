
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
    print(f"Selected compute device: {device}")

    # Whisper-specific device handling
    whisper_device = device
    if whisper_device == "mps":
        print("\n[Whisper Warning]")
        print("Whisper can be unstable with MPS on certain configurations.")
        print("Falling back to CPU mode for the transcription step to ensure stability.")
        print("Other models (like the LLM) will still use MPS if available.")
        print("To hide this warning, set COMPUTE_DEVICE=cpu in your .env file.\n")
        whisper_device = "cpu"

    print(f"Loading Whisper model (device: {whisper_device})...")
    model = whisper.load_model("base", device=whisper_device) 
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path)

    # Save transcript
    config.ensure_dirs_exist()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{Path(audio_path).stem}_{timestamp}.json"
    output_path = config.TRANSCRIPTS_DIR / output_filename
    
    # Add original audio path to the transcript JSON for diarization
    result['audio_path'] = str(Path(audio_path).resolve())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    return str(output_path)
