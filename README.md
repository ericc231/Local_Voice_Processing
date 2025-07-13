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
├── models/                 # Downloaded GGUF models
├── src/                    # Main source code
│   ├── __init__.py
│   ├── asr.py
│   ├── config.py
│   ├── data_manager.py
│   ├── diarization.py
│   ├── embedding.py
│   └── summarization.py
├── summaries/              # Generated summaries
├── transcripts/            # Raw ASR transcripts
├── .env                    # Environment variables
├── .gitignore              # Git ignore file
├── environment.yml         # Conda environment file
└── README.md               # This file
```

## Setup

1. **Create and activate the Conda environment:**

   ```bash
   # Create the environment from the file (this will install most dependencies including PyTorch, Torchaudio, etc.)
   conda env create -f environment.yml

   # Activate the new environment
   conda activate local_voice_processing

   # Install remaining Python dependencies
   pip install -r requirements.txt
   ```

   ```dotenv
   # .env
   COMPUTE_DEVICE=auto
   HUGGING_FACE_HUB_TOKEN=your_huggingface_read_token_here # Required for pyannote.audio and SpeechBrain models
   ```

2. **Configure the LLM Model:**

   Edit the `.env` file to specify the Hugging Face repository and filename for the GGUF model you wish to use. The default is Qwen2-7B-Instruct, which is recommended for mixed Chinese-English environments.


3. **First Run**: The first time you run the summarization feature, the script will automatically download the specified model into the `models/` directory. This may take some time depending on your internet connection.

## Usage

Run the main pipeline with a `.wav` audio file:

```bash
python -m src.main audio/your_audio_file.wav --summarize
```

**Note**: The input audio file must be in `.wav` format. If you have another format (e.g., `.mp3`, `.m4a`), please convert it first. A sample `ffmpeg` command is provided by the script if you use a wrong file type.

**File Deduplication and Resume**: The system calculates the SHA256 hash of the input audio file. If a file with the same hash has been previously processed (or partially processed), the system will either skip processing (if completed) or resume from the last known stage.
