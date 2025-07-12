"""
LLM-based text summarization using llama-cpp-python.
Includes automatic model downloading from Hugging Face Hub.
"""
import json
from pathlib import Path
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from . import config
from . import data_manager # Import data_manager

# Global variable to hold the loaded model
llm_model = None

def get_model_path():
    """Checks if the model exists locally, if not, downloads it."""
    repo_id = config.LLM_GGUF_REPO
    filename = config.LLM_GGUF_FILENAME
    
    if not repo_id or not filename:
        raise ValueError("LLM_GGUF_REPO and LLM_GGUF_FILENAME must be set in the .env file.")

    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=config.PROJECT_ROOT / "models", 
            local_dir_use_symlinks=False 
        )
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

def load_model():
    """Loads the LLM model, downloading it first if necessary."""
    global llm_model
    if llm_model is None:
        model_path = get_model_path()
        print(f"Loading LLM model from: {model_path}")
        llm_model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=True,
        )

def summarize_transcript(diarized_transcript_path):
    """Summarizes a transcript using the loaded local LLM."""
    load_model()

    print(f"Summarizing transcript: {diarized_transcript_path}")
    with open(diarized_transcript_path, 'r', encoding='utf-8') as f:
        diarized_transcript = json.load(f)

    # Dynamically get real names for speakers from the database
    speaker_name_map = {}
    all_speakers = data_manager.get_all_speakers()
    for speaker_record in all_speakers:
        speaker_name_map[speaker_record['temp_name']] = speaker_record['real_name']

    # Prepare full text with real names for summarization
    full_text_lines = []
    for segment in diarized_transcript['segments']:
        temp_speaker_name = segment.get('speaker', 'UNKNOWN')
        real_speaker_name = speaker_name_map.get(temp_speaker_name, temp_speaker_name) # Use real name if available
        full_text_lines.append(f"{real_speaker_name}: {segment['text'].strip()}")
    full_text = "\n".join(full_text_lines)

    # Create the prompt using the Qwen1.5 Chat format
    prompt_template = """<|im_start|>system
You are a helpful assistant. Your task is to provide a concise summary of the following conversation. The summary should be in the same language as the conversation.<|im_end|>
<|im_start|>user
Conversation:
---
{text}
---

Please provide a summary of the conversation.<|im_end|>
<|im_start|>assistant
"""
    prompt = prompt_template.format(text=full_text)

    print("Generating summary...")
    stream = llm_model(
        prompt,
        max_tokens=512,
        stop=["<|im_end|>", "<|endoftext|>"],
        echo=False,
        stream=True
    )

    summary_tokens = []
    for output in tqdm(stream, desc="Generating summary tokens"): 
        token = output['choices'][0]['text']
        summary_tokens.append(token)
    
    summary = "".join(summary_tokens).strip()

    # Save summary to file
    summary_output_dir = config.SUMMARIES_DIR
    summary_output_dir.mkdir(parents=True, exist_ok=True)
    summary_filename = Path(diarized_transcript_path).stem.replace("_diarized", "") + "_summary.txt"
    summary_path = summary_output_dir / summary_filename
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Summary saved to: {summary_path}")

    # Update transcript record in database with summary path
    original_audio_path = diarized_transcript.get('audio_path')
    if original_audio_path:
        data_manager.update_transcript_record(original_audio_path, summary_path=str(summary_path))

    return summary