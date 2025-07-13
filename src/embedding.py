

"""
Speaker embedding generation and matching using SpeechBrain.
"""
import json
import os
from tqdm import tqdm
from pathlib import Path
import torch
from speechbrain.inference.speaker import EncoderClassifier
import numpy as np
import torchaudio
from torchaudio.functional.functional import extract_segments # Direct import from functional.py

from . import config
from . import data_manager

# Global variable to hold the loaded speaker embedding model
speaker_embedding_model = None

def load_speaker_embedding_model():
    """Loads the speaker embedding model."""
    global speaker_embedding_model
    if speaker_embedding_model is None:
        device = config.get_compute_device()
        print(f"Loading Speaker Embedding Model (device: {device})...")
        
        # SpeechBrain-specific device handling
        speechbrain_device = device
        if str(speechbrain_device) == "mps":
            print("\n[SpeechBrain Warning]")
            print("SpeechBrain models can sometimes have compatibility issues with MPS.")
            print("Falling back to CPU mode for embedding to ensure stability.")
            print("To hide this warning, set COMPUTE_DEVICE=cpu in your .env file.\n")
            speechbrain_device = torch.device("cpu")

        try:
            speaker_embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb" # Local cache directory
            ).to(speechbrain_device)
        except Exception as e:
            print(f"Error loading SpeechBrain model: {e}")
            print("Please ensure you have accepted the user conditions on Hugging Face Hub for 'speechbrain/spkrec-ecapa-voxceleb'.")
            print("Visit https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb and accept the license.")
            raise

def get_embedding(audio_path=None, signal=None, sample_rate=None):
    """Extracts a speaker embedding from an audio file or a segment of it.
    Can take either audio_path or (signal, sample_rate) directly.
    Times are in seconds.
    """
    load_speaker_embedding_model()
    
    if audio_path:
        # Load the full audio file if path is provided
        signal, sample_rate = torchaudio.load(audio_path)
    elif signal is None or sample_rate is None:
        raise ValueError("Either audio_path or (signal, sample_rate) must be provided.")

    # Ensure mono and correct sample rate (SpeechBrain model expects 16kHz mono)
    if signal.shape[0] > 1: # If stereo, convert to mono
        signal = signal.mean(dim=0, keepdim=True)
    
    if sample_rate != 16000: # Resample if not 16kHz
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        signal = resampler(signal)
        sample_rate = 16000

    # Move signal to the correct device
    signal = signal.to(speaker_embedding_model.device)

    # Extract embedding
    with torch.no_grad():
        embedding = speaker_embedding_model.encode_batch(signal).squeeze().cpu().numpy()
    return embedding

def compare_embeddings(emb1, emb2):
    """Calculates cosine similarity between two embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def assign_global_speaker_id_and_save_embedding(new_embedding: np.ndarray):
    """Compares a new embedding against known global speakers.
    If a match is found, returns the existing global_speaker_id.
    Otherwise, generates a new global_speaker_id, saves the embedding, and returns the new ID.
    """
    print("Assigning global speaker ID...")
    known_speakers_records = data_manager.get_all_global_speakers()
    
    best_match_global_id = None
    highest_similarity = -1.0
    
    # Load known embeddings for comparison
    known_embeddings = {}
    for speaker_record in known_speakers_records:
        global_id = speaker_record['global_speaker_id']
        emb_path = Path(speaker_record['embedding_path'])
        if emb_path.exists():
            known_embeddings[global_id] = np.load(emb_path)
        else:
            print(f"Warning: Embedding file not found for global speaker {global_id}: {emb_path}")

    # Compare with existing embeddings
    if known_embeddings:
        for global_id, known_emb in tqdm(known_embeddings.items(), desc="Comparing with known speakers"):
            similarity = compare_embeddings(new_embedding, known_emb)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_global_id = global_id
    
    # Threshold for considering a match
    threshold = 0.75 # Adjust based on your model and data

    if highest_similarity > threshold:
        print(f"Matched with existing speaker {best_match_global_id} (Similarity: {highest_similarity:.4f})")
        return best_match_global_id
    else:
        # No strong match, create a new global speaker ID
        new_global_id = data_manager.get_next_global_speaker_id()
        embedding_filename = f"{new_global_id}.npy"
        embedding_path = config.EMBEDDINGS_DIR / embedding_filename
        np.save(embedding_path, new_embedding)
        data_manager.add_global_speaker(new_global_id, str(embedding_path))
        print(f"Created new global speaker: {new_global_id} (Similarity to best match: {highest_similarity:.4f})")
        return new_global_id

def generate_embeddings(diarized_transcript_path, audio_sha256):
    """Generates and saves speaker embeddings from a diarized transcript.
    For each speaker, it extracts an embedding and assigns a global speaker ID.
    """
    print(f"Generating embeddings for: {diarized_transcript_path}")
    
    with open(diarized_transcript_path, 'r', encoding='utf-8') as f:
        diarized_transcript = json.load(f)

    original_audio_path = diarized_transcript.get('audio_path')
    if not original_audio_path or not Path(original_audio_path).exists():
        raise FileNotFoundError(f"Original audio file not found for embedding: {original_audio_path}")

    # Group segments by temporary speaker ID (SPEAKER_00, SPEAKER_01, etc.)
    temp_speaker_segments_map = {}
    for segment in diarized_transcript['segments']:
        temp_speaker = segment['speaker']
        if temp_speaker not in temp_speaker_segments_map:
            temp_speaker_segments_map[temp_speaker] = []
        temp_speaker_segments_map[temp_speaker].append(segment)

    config.ensure_dirs_exist()
    
    # Load full audio once for all segment extractions
    full_signal, sample_rate = torchaudio.load(original_audio_path)
    
    # Ensure mono and 16kHz for segment extraction
    if full_signal.shape[0] > 1: 
        full_signal = full_signal.mean(dim=0, keepdim=True)
    if sample_rate != 16000: 
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        full_signal = resampler(full_signal)
        sample_rate = 16000

    # Map temporary speaker IDs to global speaker IDs
    temp_to_global_speaker_map = {}

    for temp_speaker in tqdm(sorted(temp_speaker_segments_map.keys()), desc="Processing temporary speakers"):
        segments = temp_speaker_segments_map[temp_speaker]
        
        # Concatenate all audio segments for the current temporary speaker
        indices = []
        for seg in segments:
            start_frame = int(seg['start'] * sample_rate)
            end_frame = int(seg['end'] * sample_rate)
            end_frame = min(end_frame, full_signal.shape[1]) # Ensure end_frame does not exceed signal length
            if start_frame < end_frame: # Only add valid segments
                indices.append((start_frame, end_frame))
        
        if indices: 
            extracted_signals = []
            for start_frame, end_frame in indices:
                extracted_signals.append(full_signal[:, start_frame:end_frame])
            
            concatenated_speaker_audio = torch.cat(extracted_signals, dim=1)
            
            # Get embedding from the concatenated audio
            current_speaker_embedding = get_embedding(audio_path=None, signal=concatenated_speaker_audio, sample_rate=sample_rate)
            
            # Assign a global speaker ID based on comparison
            global_speaker_id = assign_global_speaker_id_and_save_embedding(current_speaker_embedding)
            temp_to_global_speaker_map[temp_speaker] = global_speaker_id
        else:
            # If no segments for this temp_speaker, assign a dummy global ID or skip
            print(f"Warning: No valid audio segments found for temporary speaker {temp_speaker}. Assigning UNKNOWN_SPEAKER.")
            temp_to_global_speaker_map[temp_speaker] = "UNKNOWN_SPEAKER"

    # Update the diarized transcript JSON with global speaker IDs
    updated_diarized_segments = []
    for segment in diarized_transcript['segments']:
        temp_speaker = segment['speaker']
        segment['speaker'] = temp_to_global_speaker_map.get(temp_speaker, temp_speaker) # Replace with global ID
        updated_diarized_segments.append(segment)
    
    diarized_transcript['segments'] = updated_diarized_segments

    # Save the updated diarized transcript (with global IDs)
    output_filename = f"{audio_sha256}.json"
    output_path = config.DIARIZED_TRANSCRIPTS_DIR / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(diarized_transcript, f, ensure_ascii=False, indent=4)

    # The generate_embeddings function now also updates the diarized transcript file
    # so we return the path to the updated file.
    return str(output_path)

