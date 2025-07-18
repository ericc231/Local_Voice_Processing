
"""
ASR module using Whisper.
"""
import os
import whisper
import json
from datetime import datetime
import torch # Import torch

def transcribe_audio(audio_path, audio_sha256, initial_prompt=""):
    """Transcribes an audio file using Whisper and saves the output as JSON."""
    from . import config
    from pathlib import Path

    device = config.get_compute_device()
    print(f"Selected compute device: {device}")

    # Whisper-specific device handling
    whisper_device = device
    if str(whisper_device) == "mps": # Convert to string for comparison
        print("\n[Whisper Warning]")
        print("Whisper can be unstable with MPS on certain configurations.")
        print("Falling back to CPU mode for the transcription step to ensure stability.")
        print("Other models (like the LLM) will still use MPS if available.")
        print("To hide this warning, set COMPUTE_DEVICE=cpu in your .env file.\n")
        whisper_device = torch.device("cpu") # Set to torch.device object

    print(f"Loading Whisper model (device: {whisper_device})...")
    model = whisper.load_model("base", device=whisper_device) 
    print(f"Transcribing {audio_path}...")
    # Use the passed initial_prompt, which could be from the file or the .env config
    result = model.transcribe(audio_path, initial_prompt=initial_prompt)

    # Save transcript
    config.ensure_dirs_exist()
    output_filename = f"{audio_sha256}.json"
    output_path = config.TRANSCRIPTS_DIR / output_filename
    
    # Add original audio path to the transcript JSON for diarization
    result['audio_path'] = str(Path(audio_path).resolve())
    result['audio_sha256'] = audio_sha256
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    return str(output_path)
