
# Local Voice Processing Pipeline

This project provides a complete, local-only pipeline for processing audio files, including speech recognition, speaker diarization, voice embedding, and summarization.

## Features

- **Speech-to-Text (ASR)**: Uses OpenAI's Whisper for fast and accurate transcription. Supports CPU, CUDA, and Apple MPS.
- **Speaker Diarization**: Identifies and separates different speakers in the audio.
- **Voice Biometrics**: Creates voice embeddings for each speaker for recognition and matching.
- **AI Summarization**: Uses a local Large Language Model (LLM) to generate summaries of the transcripts.
- **Data Management**: Stores all metadata and results in a local SQLite database.
- **Privacy-Focused**: No data ever leaves your machine.

## Project Structure

```
local_voice_processing/
├── audio/                  # Input audio files
├── database/               # SQLite database
├── diarized_transcripts/   # Transcripts with speaker labels
├── embeddings/             # Speaker voice embeddings
├── scripts/                # Helper scripts
├── src/                    # Main source code
│   ├── __init__.py
│   ├── asr.py
│   ├── data_manager.py
│   ├── diarization.py
│   ├── embedding.py
│   └── summarization.py
├── summaries/              # Generated summaries
├── transcripts/            # Raw ASR transcripts
├── .env                    # Environment variables
├── environment.yml         # Conda environment file
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── setup_env.sh            # Environment setup script
```

## Setup

1. **Create and activate the environment:**

   ```bash
   bash setup_env.sh
   ```

2. **Configure your environment:**

   Edit the `.env` file to set your compute device and add your Hugging Face token if needed.

   ```dotenv
   # Options: auto, cpu, cuda, mps
   COMPUTE_DEVICE=auto
   HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

## Usage

Run the main pipeline with:

```bash
python -m src.main audio/your_audio_file.wav --summarize
```
