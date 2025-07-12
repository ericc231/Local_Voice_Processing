
"""
Data management using SQLite.
"""
import sqlite3
from . import config

def init_db():
    """Initializes the SQLite database."""
    config.ensure_dirs_exist()
    conn = sqlite3.connect(config.DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS speakers (
            id INTEGER PRIMARY KEY, 
            name TEXT, 
            embedding_path TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY, 
            audio_path TEXT, 
            transcript_path TEXT, 
            diarized_transcript_path TEXT, 
            summary_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
