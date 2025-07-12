
"""
Main script to run the voice processing pipeline.
"""
import argparse
from . import asr, diarization, embedding, summarization, data_manager, config

def main():
    config.ensure_dirs_exist()
    parser = argparse.ArgumentParser(description="Local Voice Processing Pipeline")
    parser.add_argument("audio_path", type=str, help="Path to the audio file to process (must be a .wav file).")
    parser.add_argument("--summarize", action="store_true", help="Generate a summary of the transcript.")
    args = parser.parse_args()

    # --- Input Validation ---
    if not args.audio_path.lower().endswith('.wav'):
        print(f"Error: Input file must be a .wav file.")
        print(f"Please convert your audio file to WAV format first.")
        print(f"You can use ffmpeg: ffmpeg -i '{args.audio_path}' -ar 16000 -ac 1 -c:a pcm_s16le output.wav")
        return

    # 1. ASR
    transcript_path = asr.transcribe_audio(args.audio_path)
    print(f"Transcript saved to: {transcript_path}")
    data_manager.add_transcript_record(args.audio_path, transcript_path)

    # 2. Diarization
    diarized_transcript_path = diarization.diarize_transcript(transcript_path)
    print(f"Diarized transcript saved to: {diarized_transcript_path}")
    data_manager.update_transcript_record(args.audio_path, diarized_transcript_path=diarized_transcript_path)

    # 3. Embedding
    embedding.generate_embeddings(diarized_transcript_path)
    print("Embeddings generated and saved.")

    # 4. Summarization
    if args.summarize:
        summary = summarization.summarize_transcript(diarized_transcript_path)
        print(f"Summary: {summary}")
        # Summary path is now updated within summarization.py

if __name__ == "__main__":
    main()
