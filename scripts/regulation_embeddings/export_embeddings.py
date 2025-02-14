"""Script to export regulation embeddings and metadata to JSON."""

import json
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime

def convert_datetime(dt):
    """Convert datetime objects to ISO format strings."""
    if isinstance(dt, datetime):
        return dt.isoformat()
    return dt

def export_embeddings(db_path, output_path):
    """Export embeddings and metadata from SQLite database to JSON.
    
    Args:
        db_path (str): Path to the SQLite database
        output_path (str): Path where JSON file will be saved
    """
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all records except chunk_text
    cursor.execute("""
        SELECT id, agency, title, chapter, date, chunk_index, embedding,
               section, hierarchy, created_at
        FROM regulation_chunks
    """)
    
    records = []
    for row in cursor.fetchall():
        # Convert embedding bytes to numpy array and then to list
        embedding_bytes = row[6]
        embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        record = {
            'id': row[0],
            'agency': row[1],
            'title': row[2],
            'chapter': row[3],
            'date': row[4],
            'chunk_index': row[5],
            'embedding': embedding_array.tolist(),
            'section': row[7],
            'hierarchy': row[8],
            'created_at': convert_datetime(row[9])
        }
        records.append(record)
    
    # Close database connection
    conn.close()
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(records, f, indent=2, default=convert_datetime)
    
    print(f"Exported {len(records)} records to {output_path}")

if __name__ == "__main__":
    # Assuming the script is run from the project root
    db_path = "data/db/regulation_embeddings.db"
    output_path = "data/json/regulation_embeddings_export.json"
    
    export_embeddings(db_path, output_path) 