
"""
Utility functions for the voice processing pipeline.
"""
import hashlib

def calculate_sha256(file_path: str, chunk_size: int = 8192):
    """Calculates the SHA256 hash of a given file.
    Args:
        file_path (str): The path to the file.
        chunk_size (int): The size of chunks to read the file in.
    Returns:
        str: The SHA256 hash of the file.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()
