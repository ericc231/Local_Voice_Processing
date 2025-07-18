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
    
    print(f"[DEBUG] Attempting to download model: repo_id={repo_id}, filename={filename}")

    if not repo_id or not filename:
        raise ValueError("LLM_GGUF_REPO and LLM_GGUF_FILENAME must be set in the .env file.")

    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=config.PROJECT_ROOT / "models", # Download to project's models dir
            force_download=True # Force re-download
        )
        print(f"[DEBUG] Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

def load_model():
    """Loads the LLM model, downloading it first if necessary."""
    global llm_model
    if llm_model is None:
        model_path = get_model_path()
        print(f"[DEBUG] Final model path for Llama: {model_path}")
        llm_model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=True,
        )

def summarize_transcript(diarized_transcript_path, audio_sha256):
    """Summarizes a transcript using the loaded local LLM."""
    load_model()

    print(f"Summarizing transcript: {diarized_transcript_path}")
    with open(diarized_transcript_path, 'r', encoding='utf-8') as f:
        diarized_transcript = json.load(f)

    # Dynamically get real names for speakers from the database
    speaker_name_map = {}
    all_speakers = data_manager.get_all_global_speakers()
    for speaker_record in all_speakers:
        speaker_name_map[speaker_record['global_speaker_id']] = speaker_record['real_name']

    # Prepare full text with real names for summarization
    full_text_lines = []
    for segment in diarized_transcript['segments']:
        global_speaker_id = segment.get('speaker', 'UNKNOWN_SPEAKER')
        real_speaker_name = speaker_name_map.get(global_speaker_id, global_speaker_id) # Use real name if available
        full_text_lines.append(f"{real_speaker_name}: {segment['text'].strip()}")
    full_text = "\n".join(full_text_lines)

    # Truncate full_text to fit within the model's context window
    # A safe margin is used to account for prompt template and output tokens
    max_context_tokens = llm_model.n_ctx() - 512 # Reserve 512 tokens for prompt template and output
    if llm_model.token_eos() is None: # Fallback for models without explicit EOS token
        max_context_tokens = llm_model.n_ctx() - 512

    # Encode the text to tokens to get its length
    encoded_text = llm_model.tokenize(full_text.encode("utf-8"))
    if len(encoded_text) > max_context_tokens:
        print(f"Warning: Conversation text ({len(encoded_text)} tokens) exceeds model context window. Truncating to {max_context_tokens} tokens.")
        # Decode truncated tokens back to text
        truncated_encoded_text = encoded_text[:max_context_tokens]
        full_text = llm_model.detokenize(truncated_encoded_text).decode("utf-8", errors="ignore")

    # Create the prompt using the Qwen1.5 Chat format
    prompt_template = """<|im_start|>system
You are a helpful assistant. Your task is to provide a concise summary of the following conversation.
The summary MUST be in the exact same language as the conversation. DO NOT translate the text under any circumstances. If the conversation is in Traditional Chinese, the summary MUST be in Traditional Chinese. If English, the summary MUST be in English.
<|im_end|>
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
    summary_filename = f"{audio_sha256}.txt"
    summary_path = summary_output_dir / summary_filename
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Summary saved to: {summary_path}")

    # Update transcript record in database with summary path
    original_audio_path = diarized_transcript.get('audio_path')
    if original_audio_path:
        data_manager.save_transcript_record(original_audio_path, audio_sha256, summary_path=str(summary_path), status='summarized')

    return summary