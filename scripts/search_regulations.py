#!/usr/bin/env python3
"""
Script to search regulations using a Faiss index and sentence embeddings.
Finds relevant regulation chunks based on semantic similarity to the query.

Example usage:
    python scripts/search_regulations.py "Can I destroy a national monument?"
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

SAMPLE_QUESTIONS = [
    "What are the requirements for filing a FOIA request?",
    "Can I destroy a national monument?",
    "What are the safety requirements for commercial vehicles?",
    "How are medical devices classified by the FDA?",
    "What are the rules for importing food products?",
    "How do I report research misconduct?",
    "What are the requirements for federal grant applications?",
    "How are endangered species protected?",
    "What are the regulations for drone operations?",
    "How are tribal lands managed?",
    "What are the requirements for pharmaceutical labeling?",
    "How are federal contracts awarded?",
    "What are the workplace safety requirements?",
    "How are national parks protected?",
    "What are the rules for federal student loans?",
]


class RegulationSearcher:
    def __init__(
        self, index_path: str, metadata_path: str, db_path: str, model_name: str
    ):
        self.index = self._load_faiss_index(index_path)
        self.metadata = self._load_metadata(metadata_path)
        self.db_path = db_path

        # Verify database connection and content
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM regulation_chunks")
                count = cursor.fetchone()[0]
                print(f"Connected to database. Found {count} regulation chunks")

                # Check a sample record
                cursor.execute(
                    """
                    SELECT id, chunk_text 
                    FROM regulation_chunks 
                    WHERE chunk_text IS NOT NULL 
                    LIMIT 1
                """
                )
                sample = cursor.fetchone()
                if sample:
                    print(f"Successfully retrieved sample chunk (id: {sample[0]})")
                else:
                    print("WARNING: No chunks with text found in database")
        except sqlite3.Error as e:
            raise RuntimeError(f"Database connection error: {e}")

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _load_faiss_index(index_path: str) -> faiss.IndexIDMap:
        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        try:
            index = faiss.read_index(index_path)
            print(f"Loaded Faiss index with {index.ntotal} vectors")
            return index
        except Exception as e:
            raise RuntimeError(f"Error loading Faiss index: {e}")

    @staticmethod
    def _load_metadata(metadata_path: str) -> dict:
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"Loaded metadata for {len(metadata)} records")
            return metadata
        except Exception as e:
            raise RuntimeError(f"Error loading metadata: {e}")

    def _load_chunk_text(self, chunk_id: int) -> str:
        """Load chunk text from metadata instead of database."""
        metadata = self.metadata.get(str(chunk_id))
        if metadata and "chunk_text" in metadata:
            return metadata["chunk_text"].strip()
        return "Chunk text not found"

    @staticmethod
    def _format_result(metadata: dict, score: float, chunk_text: str) -> str:
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

    def search(self, query: str, n_results: int = 5) -> list:
        """Search for relevant regulation chunks."""
        print(f"Processing query: {query}")

        try:
            # Create query embedding
            query_embedding = self.model.encode([query])[0].astype(np.float32)

            # Normalize the query embedding to unit length
            faiss.normalize_L2(query_embedding.reshape(1, -1))

            # Search the index with increased n_results to account for filtering
            search_k = min(
                n_results * 10, self.index.ntotal
            )  # Search more results initially
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1), search_k
            )

            results = []
            seen_texts = set()  # To avoid duplicate chunks

            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:  # Skip if no result found
                    continue

                # Convert distance to cosine similarity score
                similarity = 1 - (
                    distance / 2
                )  # Convert L2 distance to cosine similarity

                # Relaxed similarity threshold
                if similarity < 0.2:  # Lowered from 0.3
                    continue

                # Get metadata for this result
                result_metadata = self.metadata.get(str(idx))
                if result_metadata:
                    chunk_text = self._load_chunk_text(idx)

                    # Skip if chunk text is invalid or duplicate
                    if chunk_text == "Chunk text not found" or chunk_text in seen_texts:
                        continue

                    seen_texts.add(chunk_text)
                    results.append((result_metadata, similarity, chunk_text))

            # Sort results by similarity score in descending order
            results.sort(key=lambda x: x[1], reverse=True)

            # Trim to requested number of results
            results = results[:n_results]

            return results

        except Exception as e:
            print(f"Error during search: {str(e)}")
            import traceback

            traceback.print_exc()
            return []

    def save_results(
        self, query: str, results: list, output_path: str = "data/search_results.txt"
    ):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Search Query: {query}\n\n")
            f.write(f"Found {len(results)} results:\n\n")
            for idx, (metadata, score, chunk_text) in enumerate(results, 1):
                f.write(f"Result {idx}:\n")
                f.write(self._format_result(metadata, score, chunk_text))
                f.write("-" * 80 + "\n\n")
        print(f"Results saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search regulations using semantic similarity."
    )
    parser.add_argument(
        "query", nargs="?", type=str, help="Search query or question about regulations"
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
    parser.add_argument("--save", action="store_true", help="Save results to a file")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize searcher (loads model, index, and metadata once)
    searcher = RegulationSearcher(
        index_path=args.index,
        metadata_path=args.metadata,
        db_path=args.db,
        model_name=args.model,
    )

    # Interactive loop: use provided query or prompt the user repeatedly.
    query = args.query
    while True:
        if not query:
            query = input(
                "\nEnter your search query (press Enter for a random question, 'quit' to exit): "
            ).strip()
            if query.lower() == "quit":
                break
            if not query:
                query = random.choice(SAMPLE_QUESTIONS)
                print(f"\nUsing random question: {query}")

        try:
            results = searcher.search(query, n_results=args.num_results)
            print(f"\nSearch Query: {query}\n")
            print(f"Found {len(results)} results:\n")
            for idx, (metadata_item, score, chunk_text) in enumerate(results, 1):
                print(f"Result {idx}:")
                print(
                    RegulationSearcher._format_result(metadata_item, score, chunk_text)
                )
                print("-" * 80 + "\n")

            if args.save:
                searcher.save_results(query, results)

        except Exception as e:
            print(f"Error performing search: {e}")

        query = None  # Reset query for next iteration

    return 0


if __name__ == "__main__":
    exit(main())
