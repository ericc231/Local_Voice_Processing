
"""
Main script to run the voice processing pipeline.
"""
import argparse
from . import asr, diarization, embedding, summarization, data_manager, config

def main():
    config.ensure_dirs_exist()
    parser = argparse.ArgumentParser(description="Local Voice Processing Pipeline")
    parser.add_argument("audio_path", type=str, help="Path to the audio file to process.")
    parser.add_argument("--summarize", action="store_true", help="Generate a summary of the transcript.")
    args = parser.parse_args()

    # 1. ASR
    transcript_path = asr.transcribe_audio(args.audio_path)
    print(f"Transcript saved to: {transcript_path}")

    # 2. Diarization
    diarized_transcript_path = diarization.diarize_transcript(transcript_path)
    print(f"Diarized transcript saved to: {diarized_transcript_path}")

    # 3. Embedding
    embedding.generate_embeddings(diarized_transcript_path)
    print("Embeddings generated and saved.")

    # 4. Summarization
    if args.summarize:
        summary = summarization.summarize_transcript(diarized_transcript_path)
        print(f"Summary: {summary}")

if __name__ == "__main__":
    main()
