"""
Speaker embedding generation and matching using SpeechBrain.
"""
import json
import os
from tqdm import tqdm
from pathlib import Path
import torch
from speechbrain.pretrained import EncoderClassifier
import numpy as np
import torchaudio
from torchaudio.functional import extract_segments

from . import config

# Global variable to hold the loaded speaker embedding model
speaker_embedding_model = None

def load_speaker_embedding_model():
    """Loads the speaker embedding model."""
    global speaker_embedding_model
    if speaker_embedding_model is None:
        device = config.get_compute_device()
        print(f"Loading Speaker Embedding Model (device: {device})...")
        try:
            speaker_embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb" # Local cache directory
            ).to(device)
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

def generate_embeddings(diarized_transcript_path):
    """Generates and saves speaker embeddings from a diarized transcript.
    For each speaker, it concatenates all their segments and extracts one embedding.
    """
    print(f"Generating embeddings for: {diarized_transcript_path}")
    
    with open(diarized_transcript_path, 'r', encoding='utf-8') as f:
        diarized_transcript = json.load(f)

    original_audio_path = diarized_transcript.get('audio_path')
    if not original_audio_path or not Path(original_audio_path).exists():
        raise FileNotFoundError(f"Original audio file not found for embedding: {original_audio_path}")

    unique_speakers = sorted(list(set(segment['speaker'] for segment in diarized_transcript['segments'])))
    
    config.ensure_dirs_exist()
    
    # Group segments by speaker
    speaker_segments_map = {}
    for segment in diarized_transcript['segments']:
        speaker = segment['speaker']
        if speaker not in speaker_segments_map:
            speaker_segments_map[speaker] = []
        speaker_segments_map[speaker].append(segment)

    # Generate embedding for each speaker
    for speaker in tqdm(unique_speakers, desc="Generating speaker embeddings"):
        segments = speaker_segments_map[speaker]
        
        # Concatenate all audio segments for the current speaker
        # This is a simplified approach. For very long audios, this might be memory intensive.
        # A more robust approach might involve averaging embeddings of individual segments.
        speaker_audio_segments = []
        full_signal, sample_rate = torchaudio.load(original_audio_path)
        
        # Ensure mono and 16kHz for segment extraction
        if full_signal.shape[0] > 1: 
            full_signal = full_signal.mean(dim=0, keepdim=True)
        if sample_rate != 16000: 
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            full_signal = resampler(full_signal)
            sample_rate = 16000

        indices = []
        for seg in segments:
            start_frame = int(seg['start'] * sample_rate)
            end_frame = int(seg['end'] * sample_rate)
            indices.append(torch.tensor([start_frame, end_frame]))
        
        if indices: # Only extract if there are segments for the speaker
            # extract_segments expects a list of tensors, each [start_frame, end_frame]
            extracted_signals = extract_segments(full_signal, torch.stack(indices))
            
            # Concatenate all extracted segments for this speaker
            concatenated_speaker_audio = torch.cat(extracted_signals, dim=1)
            
            # Get embedding from the concatenated audio
            embedding = get_embedding(audio_path=None, signal=concatenated_speaker_audio, sample_rate=sample_rate) # Pass signal directly
        else:
            # If no segments for some reason, create a dummy embedding or skip
            embedding = np.zeros(512) # ECAPA-TDNN typically outputs 512-dim embeddings

        embedding_filename = f"{speaker}.npy"
        embedding_path = config.EMBEDDINGS_DIR / embedding_filename
        np.save(embedding_path, embedding)
        
        # Add/Update speaker in database
        from . import data_manager
        data_manager.add_or_update_speaker(speaker, str(embedding_path))

def compare_embeddings(emb1, emb2):
    """Calculates cosine similarity between two embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def match_speaker(new_embedding_path):
    """Matches a new embedding against known embeddings in the database.
    Returns (speaker_id, similarity) of the best match, or (None, None) if no match above threshold.
    """
    new_embedding = np.load(new_embedding_path)
    
    print("Matching speaker against known embeddings...")
    from . import data_manager
    known_speakers_records = data_manager.get_all_speakers()
    
    known_embeddings = {}
    for speaker_record in known_speakers_records:
        temp_name = speaker_record['temp_name']
        emb_path = Path(speaker_record['embedding_path'])
        if emb_path.exists():
            known_embeddings[temp_name] = np.load(emb_path)
        else:
            print(f"Warning: Embedding file not found for speaker {temp_name}: {emb_path}")

    best_match_speaker = None
    highest_similarity = -1.0
    
    if not known_embeddings:
        print("No known embeddings found for comparison.")
        return None, None

    for temp_name, known_emb in tqdm(known_embeddings.items(), desc="Comparing embeddings"):
        similarity = compare_embeddings(new_embedding, known_emb)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_speaker = temp_name
            
    # Example threshold: adjust based on your data and desired precision/recall
    # A common threshold for ECAPA-TDNN is around 0.7 to 0.8
    threshold = 0.75 
    if highest_similarity > threshold:
        print(f"Best match: {best_match_speaker} with similarity {highest_similarity:.4f}")
        return best_match_speaker, highest_similarity
    else:
        print(f"No strong match found. Highest similarity: {highest_similarity:.4f}")
        return None, highest_similarity