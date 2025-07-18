"""
Speaker diarization module using pyannote.audio.
"""
import json
from tqdm import tqdm
from pyannote.audio import Pipeline
from pathlib import Path
from . import config

# Global variable to hold the loaded diarization pipeline
diarization_pipeline = None

def load_diarization_pipeline():
    """Loads the speaker diarization pipeline."""
    global diarization_pipeline
    if diarization_pipeline is None:
        # Ensure Hugging Face token is available
        hf_token = config.HUGGING_FACE_HUB_TOKEN
        if not hf_token:
            raise ValueError("HUGGING_FACE_HUB_TOKEN is not set in .env. It's required for pyannote.audio models.")

        device = config.get_compute_device()
        print(f"Loading Pyannote Diarization Pipeline (device: {device})...")
        try:
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            ).to(device) # Move pipeline to the selected device
        except Exception as e:
            print(f"Error loading pyannote.audio pipeline: {e}")
            print("Please ensure you have accepted the user conditions on Hugging Face Hub for 'pyannote/speaker-diarization-3.1'.")
            print("Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the license.")
            raise

def diarize_transcript(transcript_path, audio_sha256):
    """Performs speaker diarization on the original audio and integrates with the transcript."""
    load_diarization_pipeline()

    print(f"Diarizing audio for transcript: {transcript_path}")
    
    # 1. Get original audio path from transcript metadata
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    original_audio_path = transcript_data.get('audio_path') 
    if not original_audio_path or not Path(original_audio_path).exists():
        raise FileNotFoundError(f"Original audio file not found for diarization: {original_audio_path}")

    # 2. Run diarization on the original audio
    print(f"Running diarization on {original_audio_path}...")
    diarization_result = diarization_pipeline(original_audio_path)

    # DEBUG: Print pyannote.audio's raw output
    unique_pyannote_speakers = set(speaker for turn, _, speaker in diarization_result.itertracks(yield_label=True))
    print(f"[DEBUG] Pyannote detected {len(unique_pyannote_speakers)} unique speakers: {sorted(list(unique_pyannote_speakers))}")
    print(f"[DEBUG] Pyannote raw diarization result:\n{diarization_result}")

    # 3. Integrate diarization results with ASR transcript
    diarized_segments = []
    
    # Convert pyannote.audio output to a more usable format
    # This creates a list of (start_time, end_time, speaker_label) tuples
    diarization_turns = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        diarization_turns.append((turn.start, turn.end, speaker))

    # Sort ASR segments by start time to ensure correct processing
    asr_segments = sorted(transcript_data['segments'], key=lambda x: x['start'])

    for asr_segment in tqdm(asr_segments, desc="Integrating diarization results"):
        asr_start = asr_segment['start']
        asr_end = asr_segment['end']
        asr_text = asr_segment['text']
        
        assigned_speaker = "UNKNOWN" # Default if no speaker found for segment
        max_overlap = 0.0

        # Find the speaker for this ASR segment based on maximum overlap with diarization results
        for diar_start, diar_end, speaker in diarization_turns:
            overlap_start = max(asr_start, diar_start)
            overlap_end = min(asr_end, diar_end)
            
            current_overlap = max(0.0, overlap_end - overlap_start)
            
            if current_overlap > max_overlap:
                max_overlap = current_overlap
                assigned_speaker = speaker

        diarized_segments.append({
            "start": asr_start,
            "end": asr_end,
            "text": asr_text,
            "speaker": assigned_speaker
        })

    # Create a new JSON structure for the diarized transcript
    diarized_transcript_data = {
        "audio_path": original_audio_path,
        "audio_sha256": audio_sha256,
        "segments": diarized_segments,
        "diarization_info": str(diarization_result) # Store raw diarization output for debugging
    }

    # Save diarized transcript
    output_filename = f"{audio_sha256}.json"
    output_path = config.DIARIZED_TRANSCRIPTS_DIR / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(diarized_transcript_data, f, ensure_ascii=False, indent=4)

    return str(output_path)