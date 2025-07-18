
"""
Data management using SQLite.
"""
import sqlite3
from . import config
from pathlib import Path

def get_db_connection():
    """Establishes and returns a database connection."""
    config.ensure_dirs_exist()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    return conn

def init_db():
    """Initializes the SQLite database schema.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS speakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            global_speaker_id TEXT UNIQUE, -- Consistent ID across all audios (e.g., GLOBAL_SPEAKER_001)
            real_name TEXT,                -- User-defined name (e.g., Eric, Alice)
            embedding_path TEXT UNIQUE
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_path TEXT UNIQUE, -- Original audio file path
            audio_sha256 TEXT UNIQUE, -- SHA256 hash of the audio file
            transcript_path TEXT,
            diarized_transcript_path TEXT,
            summary_path TEXT,
            pdf_path TEXT, -- Path to the generated PDF output
            status TEXT DEFAULT 'pending', -- e.g., 'pending', 'transcribed', 'diarized', 'embedded', 'summarized', 'completed'
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_next_global_speaker_id():
    """Generates the next available global speaker ID (e.g., GLOBAL_SPEAKER_001)."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT MAX(CAST(SUBSTR(global_speaker_id, 16) AS INTEGER)) FROM speakers WHERE global_speaker_id LIKE 'GLOBAL_SPEAKER_%'")
    max_id = c.fetchone()[0]
    conn.close()
    if max_id is None:
        return "GLOBAL_SPEAKER_001"
    else:
        return f"GLOBAL_SPEAKER_{max_id + 1:03d}"

def add_global_speaker(global_speaker_id: str, embedding_path: str):
    """Adds a new global speaker to the database.
    Initializes real_name to global_speaker_id.
    """
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO speakers (global_speaker_id, real_name, embedding_path) VALUES (?, ?, ?)", 
                  (global_speaker_id, global_speaker_id, embedding_path))
        conn.commit()
        print(f"Added new global speaker: {global_speaker_id}")
    except sqlite3.IntegrityError:
        # If global_speaker_id already exists, do nothing (it means it was already added by another process/run)
        print(f"Warning: Global speaker {global_speaker_id} already exists. Skipping add (expected behavior).")
    finally:
        conn.close()

def update_global_speaker_embedding_path(global_speaker_id: str, embedding_path: str):
    """Updates an existing global speaker's embedding path."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE speakers SET embedding_path = ? WHERE global_speaker_id = ?", 
              (embedding_path, global_speaker_id))
    conn.commit()
    conn.close()
    print(f"Updated embedding path for global speaker: {global_speaker_id}")

def rename_speaker(global_speaker_id: str, new_real_name: str):
    """Updates the real_name for a given global speaker ID."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE speakers SET real_name = ? WHERE global_speaker_id = ?", (new_real_name, global_speaker_id))
    conn.commit()
    conn.close()
    print(f"Renamed global speaker '{global_speaker_id}' to '{new_real_name}'.")

def get_speaker_by_global_id(global_speaker_id: str):
    """Retrieves a speaker record by their global speaker ID."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM speakers WHERE global_speaker_id = ?", (global_speaker_id,))
    speaker = c.fetchone()
    conn.close()
    return speaker

def get_all_global_speakers():
    """Retrieves all global speakers from the database."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM speakers")
    speakers = c.fetchall()
    conn.close()
    return speakers

def get_transcript_by_sha256(audio_sha256: str):
    """Retrieves a transcript record by its SHA256 hash."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM transcripts WHERE audio_sha256 = ?", (audio_sha256,))
    record = c.fetchone()
    conn.close()
    return record

def save_transcript_record(audio_path: str, audio_sha256: str, 
                           transcript_path: str = None, 
                           diarized_transcript_path: str = None, 
                           summary_path: str = None,
                           pdf_path: str = None, # Add new parameter
                           status: str = None):
    """Inserts a new transcript record or updates an existing one based on audio_sha256."""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Check if record already exists by SHA256
    c.execute("SELECT id FROM transcripts WHERE audio_sha256 = ?", (audio_sha256,))
    existing_record = c.fetchone()

    if existing_record:
        # Update existing record
        updates = []
        params = []
        if transcript_path:
            updates.append("transcript_path = ?")
            params.append(transcript_path)
        if diarized_transcript_path:
            updates.append("diarized_transcript_path = ?")
            params.append(diarized_transcript_path)
        if summary_path:
            updates.append("summary_path = ?")
            params.append(summary_path)
        if pdf_path:
            updates.append("pdf_path = ?")
            params.append(pdf_path)
        if status:
            updates.append("status = ?")
            params.append(status)
        
        if updates:
            params.append(audio_sha256)
            query = f"UPDATE transcripts SET {', '.join(updates)} WHERE audio_sha256 = ?"
            c.execute(query, params)
            conn.commit()
            print(f"Updated transcript record for {Path(audio_path).name} (SHA256: {audio_sha256[:8]})")
    else:
        # Insert new record
        c.execute("INSERT INTO transcripts (audio_path, audio_sha256, transcript_path, diarized_transcript_path, summary_path, pdf_path, status) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                  (audio_path, audio_sha256, transcript_path, diarized_transcript_path, summary_path, pdf_path, status if status else 'pending'))
        conn.commit()
        print(f"Added new transcript record for {Path(audio_path).name} (SHA256: {audio_sha256[:8]})")
    
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized.")
