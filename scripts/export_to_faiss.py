#!/usr/bin/env python3
"""
Script to transfer vector embeddings and metadata (excluding chunk text)
from the SQLite database (regulation_embeddings.db) to a local Faiss persistent storage.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np


def export_to_faiss(db_path: str, index_out_path: str, metadata_out_path: str) -> None:
    """Export embeddings and metadata from SQLite database to Faiss persistent storage."""
    # Create output directories if they don't exist
    Path(index_out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_out_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query records including metadata fields
    query = """
        SELECT id, agency, title, chapter, date, chunk_index, embedding, section, 
               hierarchy, created_at, chunk_text
        FROM regulation_chunks
        WHERE embedding IS NOT NULL AND chunk_text IS NOT NULL
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    if not rows:
        print("No records found in the database.")
        return

    embeddings_list = []
    ids = []
    metadata_mapping = {}

    # Process each database record
    for row in rows:
        try:
            row_id = row[0]
            embedding_bytes = row[6]

            # Convert embedding bytes into numpy array (now 1536 dimensions)
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

            # Skip invalid embeddings
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                continue

            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            embedding = embedding / norm

            embeddings_list.append(embedding)
            ids.append(row_id)

            # Save metadata mapping including enriched fields
            metadata_mapping[str(row_id)] = {
                "agency": row[1],
                "title": row[2],
                "chapter": row[3],
                "date": row[4],
                "chunk_index": row[5],
                "section": row[7],
                "hierarchy": row[8],
                "created_at": row[9],
                "chunk_text": row[10],
            }

        except Exception as e:
            print(f"Error processing row {row_id}: {e}")
            continue

    # Create FAISS index with correct dimension
    embeddings_np = np.vstack(embeddings_list)
    dim = embeddings_np.shape[1]  # Should be 1536
    index = faiss.IndexFlatL2(dim)
    index_id = faiss.IndexIDMap(index)

    # Add vectors
    ids_np = np.array(ids, dtype=np.int64)
    index_id.add_with_ids(embeddings_np, ids_np)

    # Save index and metadata
    faiss.write_index(index_id, index_out_path)
    with open(metadata_out_path, "w") as f:
        json.dump(metadata_mapping, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Export embeddings and metadata from regulation_embeddings.db to Faiss persistent storage."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/db/regulation_embeddings.db",
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--index_out",
        type=str,
        default="data/faiss/regulation_index.faiss",
        help="Output path for the Faiss index file.",
    )
    parser.add_argument(
        "--metadata_out",
        type=str,
        default="data/faiss/regulation_metadata.json",
        help="Output path for the metadata JSON file.",
    )
    args = parser.parse_args()

    export_to_faiss(args.db, args.index_out, args.metadata_out)


if __name__ == "__main__":
    main()
