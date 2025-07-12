
"""
LLM-based text summarization.
"""
import json

import os
import config

def summarize_transcript(diarized_transcript_path):
    """Summarizes a transcript using a local LLM."""
    from . import config
    # TODO: When implementing, use config.get_compute_device() to move models to the correct device.
    # example: model.to(config.get_compute_device())
    print(f"Summarizing transcript: {diarized_transcript_path}")
    # Placeholder for summarization using a local LLM (e.g., from Hugging Face)
    with open(diarized_transcript_path, 'r', encoding='utf-8') as f:
        diarized_transcript = json.load(f)

    full_text = " ".join(segment['text'] for segment in diarized_transcript['segments'])

    # Placeholder for LLM call
    summary = f"This is a summary of the transcript: {full_text[:100]}..."

    # Persist the summary to a file
    os.makedirs(config.SUMMARIES_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(diarized_transcript_path))[0]
    summary_path = os.path.join(config.SUMMARIES_DIR, f"{base_name}_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)

    return summary_path
