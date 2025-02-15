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
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query records excluding chunk_text
    query = """
        SELECT id, agency, title, chapter, date, chunk_index, embedding, section, hierarchy, created_at
        FROM regulation_chunks
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
        row_id = row[0]
        agency = row[1]
        title = row[2]
        chapter = row[3]
        date = row[4]
        chunk_index = row[5]
        embedding_bytes = row[6]
        section = row[7]
        hierarchy = row[8]
        created_at = row[9]

        # Convert embedding bytes into numpy array (dtype float32)
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        embeddings_list.append(embedding)
        ids.append(row_id)

        # Save metadata (excluding the chunk text) keyed by row id
        metadata_mapping[row_id] = {
            "agency": agency,
            "title": title,
            "chapter": chapter,
            "date": date,
            "chunk_index": chunk_index,
            "section": section,
            "hierarchy": hierarchy,
            "created_at": created_at,
        }

    # Create a numpy matrix of embeddings with shape (N, dim)
    embeddings_np = np.vstack(embeddings_list)
    dim = embeddings_np.shape[1]
    print(f"Found {len(ids)} records with embedding dimension {dim}.")

    # Build a Faiss index using IndexFlatL2 for Euclidean distance
    index = faiss.IndexFlatL2(dim)
    # Wrap with IndexIDMap to store the association between embeddings and their IDs
    index_id = faiss.IndexIDMap(index)

    # Make sure the IDs are np.int64
    ids_np = np.array(ids, dtype=np.int64)
    index_id.add_with_ids(embeddings_np, ids_np)

    # Ensure the output directory exists for the index file
    os.makedirs(os.path.dirname(index_out_path), exist_ok=True)
    # Save the Faiss index to disk
    faiss.write_index(index_id, index_out_path)
    print(f"Faiss index saved to: {index_out_path}")

    # Ensure the output directory exists for metadata JSON
    os.makedirs(os.path.dirname(metadata_out_path), exist_ok=True)
    # Save the metadata mapping to a JSON file
    with open(metadata_out_path, "w") as f:
        json.dump(metadata_mapping, f, indent=2)
    print(f"Metadata saved to: {metadata_out_path}")

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
