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
import warnings
from pathlib import Path
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Add warning filter at the top of the file
warnings.filterwarnings(
    "ignore", message=".*Torch was not compiled with flash attention.*"
)

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
    "What are the requirements for exporting goods?",
    "How are hazardous materials transported?",
    "What are the rules for using pesticides?",
    "How are water quality standards set?",
    "What are the requirements for using pesticides?",
    "How are water quality standards set?",
]


class RegulationSearcher:
    def __init__(
        self, index_path: str, metadata_path: str, db_path: str, model_name: str
    ):
        self.index = self._configure_index(self._load_faiss_index(index_path))
        self.metadata = self._load_metadata(metadata_path)
        self.db_path = db_path
        self._cache = {}

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

    @lru_cache(maxsize=1000)
    def _load_chunk_text(self, chunk_id: int) -> str:
        """Load chunk text with caching."""
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

    def search(self, query: str, n_results: int = 5, batch_size: int = 32) -> list:
        """Search for relevant regulation chunks."""
        print(f"Processing query: {query}")

        try:
            # Create query embedding - support for batch queries
            if isinstance(query, str):
                queries = [query]
            else:
                queries = query

            # Process in batches for better performance
            embeddings = []
            for i in range(0, len(queries), batch_size):
                batch = queries[i : i + batch_size]
                batch_embeddings = self.model.encode(batch).astype(np.float32)
                embeddings.append(batch_embeddings)

            query_embedding = np.vstack(embeddings)

            # Normalize embeddings
            faiss.normalize_L2(query_embedding)

            # Use inner product (cosine similarity) directly instead of L2 distance conversion
            distances, indices = self.index.search(query_embedding, n_results * 2)

            results = []
            seen_texts = set()

            # Process results more efficiently
            for batch_distances, batch_indices in zip(distances, indices):
                batch_results = []

                for distance, idx in zip(batch_distances, batch_indices):
                    if idx == -1 or distance < 0.2:  # Early filtering
                        continue

                    result_metadata = self.metadata.get(str(idx))
                    if not result_metadata:
                        continue

                    chunk_text = self._load_chunk_text(idx)
                    if chunk_text == "Chunk text not found" or chunk_text in seen_texts:
                        continue

                    seen_texts.add(chunk_text)
                    batch_results.append((result_metadata, distance, chunk_text))

                # Sort and trim results per batch
                batch_results.sort(key=lambda x: x[1], reverse=True)
                results.extend(batch_results[:n_results])

            return results[:n_results]

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

    def _configure_index(self, index: faiss.IndexIDMap) -> faiss.IndexIDMap:
        """Configure Faiss index for optimal performance."""
        # Enable internal multithreading
        faiss.omp_set_num_threads(4)
        return index


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
