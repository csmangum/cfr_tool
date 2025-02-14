#!/usr/bin/env python3
"""
Script to search regulations using a Faiss index and sentence embeddings.
Finds relevant regulation chunks based on semantic similarity to the query.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_faiss_index(index_path: str) -> faiss.IndexIDMap:
    """Load the Faiss index from disk."""
    try:
        index = faiss.read_index(index_path)
        print(f"Loaded Faiss index with {index.ntotal} vectors")
        return index
    except Exception as e:
        raise RuntimeError(f"Error loading Faiss index: {e}")


def load_metadata(metadata_path: str) -> dict:
    """Load the metadata mapping from JSON file."""
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"Loaded metadata for {len(metadata)} records")
        return metadata
    except Exception as e:
        raise RuntimeError(f"Error loading metadata: {e}")


def load_chunk_text(db_path: str, chunk_id: int) -> str:
    """Load the chunk text from the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Convert chunk_id to integer since Faiss might return it as a different type
        chunk_id = int(chunk_id)

        cursor.execute(
            "SELECT chunk_text FROM regulation_chunks WHERE id = ?", (chunk_id,)
        )
        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            return result[0].strip()
        return "Chunk text not found"
    except Exception as e:
        print(f"Error retrieving chunk text for id {chunk_id}: {e}")
        return f"Error retrieving chunk text: {e}"


def format_result(metadata: dict, score: float, chunk_text: str) -> str:
    """Format a search result for display."""
    agency = metadata.get("agency", "Unknown Agency")
    title = metadata.get("title", "Unknown Title")
    chapter = metadata.get("chapter", "Unknown Chapter")
    section = metadata.get("section", "N/A")
    date = metadata.get("date", "Unknown Date")

    return (
        f"Score: {score:.3f}\n"
        f"Agency: {agency}\n"
        f"Title {title}, Chapter {chapter}, Section {section}\n"
        f"Date: {date}\n\n"
        f"Text:\n{chunk_text}\n"
    )


def search_regulations(
    query: str,
    index_path: str,
    metadata_path: str,
    db_path: str = "data/db/regulation_embeddings.db",
    model_name: str = "all-MiniLM-L6-v2",
    n_results: int = 5,
) -> list:
    """
    Search for regulations similar to the query.

    Args:
        query: Search query text
        index_path: Path to Faiss index file
        metadata_path: Path to metadata JSON file
        db_path: Path to SQLite database containing chunk text
        model_name: Name of the sentence transformer model to use
        n_results: Number of results to return

    Returns:
        List of tuples containing (metadata, similarity_score, chunk_text)
    """
    # Load the index and metadata
    index = load_faiss_index(index_path)
    metadata = load_metadata(metadata_path)

    # Verify database exists
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    # Load the embedding model
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Create query embedding
    query_embedding = model.encode([query])[0]
    query_embedding = query_embedding.astype(np.float32)

    # Normalize the query embedding for cosine similarity
    faiss.normalize_L2(query_embedding.reshape(1, -1))

    # Search the index
    distances, indices = index.search(query_embedding.reshape(1, -1), n_results)

    # Format results
    results = []
    for idx, (distance, index) in enumerate(zip(distances[0], indices[0]), 1):
        if index == -1:  # Faiss returns -1 for padding when there are fewer results
            continue

        # Convert distance to similarity score (1 - normalized_distance)
        similarity = 1 - (
            distance / 2
        )  # Assuming normalized vectors, distance ranges from 0 to 2

        # Get metadata for this result
        result_metadata = metadata.get(str(index))  # JSON keys are strings
        if result_metadata:
            print(f"Retrieving chunk text for id: {index}")  # Debug print
            chunk_text = load_chunk_text(db_path, index)
            if chunk_text == "Chunk text not found":
                print(f"No chunk text found for id: {index}")  # Debug print
            results.append((result_metadata, similarity, chunk_text))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Search regulations using semantic similarity."
    )
    parser.add_argument(
        "query", type=str, help="Search query or question about regulations"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="data/faiss/regulation_index.faiss",
        help="Path to Faiss index file",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/faiss/regulation_metadata.json",
        help="Path to metadata JSON file",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/db/regulation_embeddings.db",
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the sentence transformer model to use",
    )
    parser.add_argument(
        "--num-results", type=int, default=5, help="Number of results to return"
    )
    args = parser.parse_args()

    try:
        # Perform the search
        results = search_regulations(
            args.query, args.index, args.metadata, args.db, args.model, args.num_results
        )

        # Print results
        print(f"\nSearch Query: {args.query}\n")
        print(f"Found {len(results)} results:\n")

        for idx, (metadata, score, chunk_text) in enumerate(results, 1):
            print(f"Result {idx}:")
            print(format_result(metadata, score, chunk_text))
            print("-" * 80 + "\n")

    except Exception as e:
        print(f"Error performing search: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
