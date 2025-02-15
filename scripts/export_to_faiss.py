#!/usr/bin/env python3
"""
Script to transfer vector embeddings and metadata (excluding chunk text)
from the SQLite database (regulation_embeddings.db) to a local Faiss persistent storage.
"""

import argparse
import json
import os
import sqlite3

import faiss
import numpy as np


def export_to_faiss(db_path: str, index_out_path: str, metadata_out_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query records including chunk_text
    query = """
        SELECT id, agency, title, chapter, date, chunk_index, embedding, section, hierarchy, created_at, chunk_text
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
    skipped = 0

    # Process each database record
    for i, row in enumerate(rows):
        try:
            row_id = row[0]
            embedding_bytes = row[6]
            chunk_text = row[10]

            if not embedding_bytes or not chunk_text:
                skipped += 1
                continue

            # Convert embedding bytes into numpy array
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

            # Skip invalid embeddings
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                skipped += 1
                continue

            # Normalize the embedding to unit length
            norm = np.linalg.norm(embedding)
            if norm == 0:
                skipped += 1
                continue

            embedding = embedding / norm

            embeddings_list.append(embedding)
            ids.append(row_id)

            # Save metadata mapping including chunk_text
            metadata_mapping[str(row_id)] = {
                "agency": row[1],
                "title": row[2],
                "chapter": row[3],
                "date": row[4],
                "chunk_index": row[5],
                "section": row[7],
                "hierarchy": row[8],
                "created_at": row[9],
                "chunk_text": chunk_text,
            }

        except Exception as e:
            skipped += 1
            continue

    # Create a numpy matrix of embeddings
    embeddings_np = np.vstack(embeddings_list)
    dim = embeddings_np.shape[1]

    # Build an L2-normalized index for cosine similarity search
    index = faiss.IndexFlatL2(dim)
    index_id = faiss.IndexIDMap(index)

    # Add vectors
    ids_np = np.array(ids, dtype=np.int64)
    index_id.add_with_ids(embeddings_np, ids_np)

    # Save the index and metadata
    os.makedirs(os.path.dirname(index_out_path), exist_ok=True)
    faiss.write_index(index_id, index_out_path)

    os.makedirs(os.path.dirname(metadata_out_path), exist_ok=True)
    with open(metadata_out_path, "w") as f:
        json.dump(metadata_mapping, f, indent=2)

    conn.close()


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
