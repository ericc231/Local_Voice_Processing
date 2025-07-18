
"""
Main script to run the voice processing pipeline.
"""
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from . import diarization, embedding, summarization, data_manager, config, utils, whisper_asr as asr, pdf_output

# Define processing stage order for robust status checking
PROCESSING_STAGES = {
    'pending': 0,
    'transcribed': 1,
    'diarized': 2,
    'embedded': 3,
    'summarized': 4,
    'pdf_generated': 5, # New stage
    'completed': 6
}

def main():
    config.ensure_dirs_exist()
    data_manager.init_db() # Initialize database tables
    parser = argparse.ArgumentParser(description="Local Voice Processing Pipeline")
    parser.add_argument("audio_path", type=str, help="Path to the audio file to process (must be a .wav file).")
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to a file containing initial prompt keywords, one per line.")
    args = parser.parse_args()

    # --- Input Validation ---
    if not args.audio_path.lower().endswith('.wav'):
        print(f"Error: Input file must be a .wav file.")
        print(f"Please convert your audio file to WAV format first.")
        print(f"You can use ffmpeg: ffmpeg -i '{args.audio_path}' -ar 16000 -ac 1 -c:a pcm_s16le output.wav")
        return

    # --- Calculate SHA256 and Check Status ---
    audio_sha256 = utils.calculate_sha256(args.audio_path)
    print(f"Audio SHA256: {audio_sha256}")

    record = data_manager.get_transcript_by_sha256(audio_sha256)
    
    if record is None:
        # This is a new audio file, save initial record with 'pending' status
        data_manager.save_transcript_record(args.audio_path, audio_sha256, status='pending')
        record = data_manager.get_transcript_by_sha256(audio_sha256) # Fetch the newly created record
        print(f"Starting new processing for {Path(args.audio_path).name} (SHA256: {audio_sha256[:8]}).")
    elif record['status'] == 'completed':
        print(f"Audio file {Path(args.audio_path).name} (SHA256: {audio_sha256[:8]}) has already been processed and completed. Skipping.")
        return
    else:
        print(f"Resuming processing for {Path(args.audio_path).name} (SHA256: {audio_sha256[:8]}). Current status: {record['status']}.")

    current_stage_index = PROCESSING_STAGES.get(record['status'], 0)

    # --- Prepare Initial Prompt ---
    initial_prompt_str = config.WHISPER_INITIAL_PROMPT
    if args.prompt_file:
        print(f"Loading initial prompt from file: {args.prompt_file}")
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                # Read lines, strip whitespace, filter out empty lines, and join with a comma and space
                prompt_keywords = [line.strip() for line in f if line.strip()]
                initial_prompt_str = ", ".join(prompt_keywords)
                print(f"Using prompt: {initial_prompt_str}")
        except FileNotFoundError:
            print(f"Warning: Prompt file not found at {args.prompt_file}. Continuing without a file-based prompt.")
        except Exception as e:
            print(f"Warning: Error reading prompt file {args.prompt_file}: {e}. Continuing without a file-based prompt.")

    # 1. ASR
    transcript_path = record['transcript_path']
    if current_stage_index < PROCESSING_STAGES['transcribed'] or not (transcript_path and Path(transcript_path).exists()):
        print(f"Starting ASR for {Path(args.audio_path).name}...")
        transcript_path = asr.transcribe_audio(args.audio_path, audio_sha256, initial_prompt=initial_prompt_str)
        print(f"Transcript saved to: {transcript_path}")
        data_manager.save_transcript_record(args.audio_path, audio_sha256, transcript_path=transcript_path, status='transcribed')
        record = data_manager.get_transcript_by_sha256(audio_sha256) # Re-fetch record
        current_stage_index = PROCESSING_STAGES['transcribed']
    else:
        print(f"ASR already completed for {Path(args.audio_path).name}. Skipping.")

    # 2. Diarization
    diarized_transcript_path = record['diarized_transcript_path']
    if current_stage_index < PROCESSING_STAGES['diarized'] or not (diarized_transcript_path and Path(diarized_transcript_path).exists()):
        print(f"Starting Diarization for {Path(args.audio_path).name}...")
        # Use the corrected transcript for diarization
        diarized_transcript_path = diarization.diarize_transcript(transcript_path, audio_sha256)
        print(f"Diarized transcript saved to: {diarized_transcript_path}")
        data_manager.save_transcript_record(args.audio_path, audio_sha256, diarized_transcript_path=diarized_transcript_path, status='diarized')
        record = data_manager.get_transcript_by_sha256(audio_sha256) # Re-fetch record
        current_stage_index = PROCESSING_STAGES['diarized']
    else:
        print(f"Diarization already completed for {Path(args.audio_path).name}. Skipping.")

    # 3. Embedding
    if current_stage_index < PROCESSING_STAGES['embedded']:
        print(f"Starting Embedding for {Path(args.audio_path).name}...")
        # generate_embeddings now returns the path to the updated diarized transcript
        diarized_transcript_path = embedding.generate_embeddings(diarized_transcript_path, audio_sha256)
        print("Embeddings generated and saved.")
        data_manager.save_transcript_record(args.audio_path, audio_sha256, diarized_transcript_path=diarized_transcript_path, status='embedded')
        record = data_manager.get_transcript_by_sha256(audio_sha256) # Re-fetch record
        current_stage_index = PROCESSING_STAGES['embedded']
    else:
        print(f"Embedding already completed for {Path(args.audio_path).name}. Skipping.")

    # 4. Summarization
    summary_path = record['summary_path']
    if current_stage_index < PROCESSING_STAGES['summarized'] or not (summary_path and Path(summary_path).exists()):
        print(f"Starting Summarization for {Path(args.audio_path).name}...")
        summary = summarization.summarize_transcript(diarized_transcript_path, audio_sha256)
        print(f"Summary: {summary}")
        # Summary path is now updated within summarization.py
        data_manager.save_transcript_record(args.audio_path, audio_sha256, summary_path=summary_path, status='summarized') # Explicitly save status
        current_stage_index = PROCESSING_STAGES['summarized']
    else:
        print(f"Summarization already completed for {Path(args.audio_path).name}. Skipping.")
    
    # 5. PDF Output
    pdf_path = record['pdf_path']
    if current_stage_index < PROCESSING_STAGES['pdf_generated'] or not (pdf_path and Path(pdf_path).exists()):
        print(f"Starting PDF generation for {Path(args.audio_path).name}...")
        pdf_path = pdf_output.generate_pdf(diarized_transcript_path, summary_path, audio_sha256)
        print(f"PDF saved to: {pdf_path}")
        data_manager.save_transcript_record(args.audio_path, audio_sha256, pdf_path=pdf_path, status='pdf_generated')
        record = data_manager.get_transcript_by_sha256(audio_sha256) # Re-fetch record
        current_stage_index = PROCESSING_STAGES['pdf_generated']
    else:
        print(f"PDF generation already completed for {Path(args.audio_path).name}. Skipping.")

    data_manager.save_transcript_record(args.audio_path, audio_sha256, status='completed')
    print(f"Processing for {Path(args.audio_path).name} (SHA256: {audio_sha256[:8]}) completed.")

if __name__ == "__main__":
    main()
