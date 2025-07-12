
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
    Drops and recreates tables for schema updates in development.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Drop tables for development schema updates
    c.execute("DROP TABLE IF EXISTS speakers")
    c.execute("DROP TABLE IF EXISTS transcripts")

    c.execute('''
        CREATE TABLE IF NOT EXISTS speakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temp_name TEXT UNIQUE, -- The temporary ID like SPEAKER_00
            real_name TEXT,        -- The user-defined real name
            embedding_path TEXT UNIQUE
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_path TEXT UNIQUE,
            transcript_path TEXT,
            diarized_transcript_path TEXT,
            summary_path TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_or_update_speaker(temp_name: str, embedding_path: str):
    """Adds a new speaker or updates an existing one's embedding path.
    Initializes real_name to temp_name if new.
    """
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO speakers (temp_name, real_name, embedding_path) VALUES (?, ?, ?)", 
                  (temp_name, temp_name, embedding_path))
        conn.commit()
        print(f"Added new speaker: {temp_name}")
    except sqlite3.IntegrityError:
        # If temp_name already exists, update the embedding path
        c.execute("UPDATE speakers SET embedding_path = ? WHERE temp_name = ?", 
                  (embedding_path, temp_name))
        conn.commit()
        print(f"Updated embedding path for speaker: {temp_name}")
    finally:
        conn.close()

def rename_speaker(temp_name: str, new_real_name: str):
    """Updates the real_name for a given temporary speaker name."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE speakers SET real_name = ? WHERE temp_name = ?", (new_real_name, temp_name))
    conn.commit()
    conn.close()
    print(f"Renamed speaker '{temp_name}' to '{new_real_name}'.")

def get_speaker_by_temp_name(temp_name: str):
    """Retrieves a speaker record by their temporary name."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM speakers WHERE temp_name = ?", (temp_name,))
    speaker = c.fetchone()
    conn.close()
    return speaker

def get_all_speakers():
    """Retrieves all speakers from the database."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM speakers")
    speakers = c.fetchall()
    conn.close()
    return speakers

def add_transcript_record(audio_path: str, transcript_path: str):
    """Adds a new transcript record."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO transcripts (audio_path, transcript_path) VALUES (?, ?)", (audio_path, transcript_path))
    conn.commit()
    record_id = c.lastrowid
    conn.close()
    print(f"Added transcript record for {Path(audio_path).name} with ID: {record_id}")
    return record_id

def update_transcript_record(audio_path: str, diarized_transcript_path: str = None, summary_path: str = None):
    """Updates an existing transcript record."""
    conn = get_db_connection()
    c = conn.cursor()
    updates = []
    params = []
    if diarized_transcript_path:
        updates.append("diarized_transcript_path = ?")
        params.append(diarized_transcript_path)
    if summary_path:
        updates.append("summary_path = ?")
        params.append(summary_path)
    
    if updates:
        params.append(audio_path)
        query = f"UPDATE transcripts SET {', '.join(updates)} WHERE audio_path = ?"
        c.execute(query, params)
        conn.commit()
        print(f"Updated transcript record for {Path(audio_path).name}")
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized.")
