
"""
Module for generating PDF output of diarized transcripts and summaries.
"""
import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import timedelta

from . import config, data_manager

def format_timestamp(seconds):
    """Formats seconds into HH:MM:SS.ms"""
    if seconds is None:
        return "00:00:00.000"
    
    total_milliseconds = int(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def generate_pdf(diarized_transcript_path: str, summary_path: str, audio_sha256: str) -> str:
    """
    Generates a PDF from a diarized transcript and a summary.

    Args:
        diarized_transcript_path: Path to the diarized transcript JSON file.
        summary_path: Path to the summary text file.
        audio_sha256: SHA256 hash of the audio, used for naming the output file.

    Returns:
        The path to the generated PDF file.
    """
    print("Generating PDF output...")

    config.ensure_dirs_exist()
    output_filename = f"{audio_sha256}.pdf"
    output_path = config.PDF_OUTPUT_DIR / output_filename

    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add Title
    story.append(Paragraph("Transcript and Summary", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # Add Summary
    story.append(Paragraph("Summary:", styles['h2']))
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_content = f.read()
        story.append(Paragraph(summary_content, styles['Normal']))
    except FileNotFoundError:
        story.append(Paragraph("Summary not found.", styles['Normal']))
    story.append(Spacer(1, 0.4 * inch))

    # Add Transcript
    story.append(Paragraph("Transcript:", styles['h2']))
    try:
        with open(diarized_transcript_path, 'r', encoding='utf-8') as f:
            diarized_data = json.load(f)

        # Dynamically get real names for speakers from the database
        speaker_name_map = {}
        all_speakers = data_manager.get_all_global_speakers()
        for speaker_record in all_speakers:
            speaker_name_map[speaker_record['global_speaker_id']] = speaker_record['real_name']

        for segment in diarized_data.get('segments', []):
            speaker_id = segment.get('speaker', 'UNKNOWN_SPEAKER')
            real_speaker_name = speaker_name_map.get(speaker_id, speaker_id)
            start_time = format_timestamp(segment.get('start'))
            
            # Format: Speaker (HH:MM:SS.ms): Text
            transcript_line = f"<b>{real_speaker_name}</b> ({start_time}): {segment['text'].strip()}"
            story.append(Paragraph(transcript_line, styles['Normal']))
            story.append(Spacer(1, 0.05 * inch)) # Small space between lines

    except (FileNotFoundError, json.JSONDecodeError) as e:
        story.append(Paragraph(f"Error loading transcript: {e}", styles['Normal']))
    
    doc.build(story)
    print(f"PDF generated successfully at: {output_path}")
    return str(output_path)
