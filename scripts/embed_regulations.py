"""
Regulation Text Embedder

This script processes regulation text files from agencies, chunks them into meaningful segments,
and creates embeddings for similarity search. It stores the embeddings and metadata in a SQLite 
database for later retrieval and searching.

The script handles:
- Reading regulation text files from agency directories
- Chunking documents into meaningful segments
- Creating embeddings using sentence-transformers
- Storing embeddings and metadata in SQLite
"""

import argparse
import logging
import os
import re
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Suppress various warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Update existing environment settings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("FAISS_CPU_ONLY", "1")
os.environ["TORCH_ALLOW_TF32"] = "1"  # Better performance on Ampere+ GPUs

import numpy as np
from regulation_embeddings.config import Config
from regulation_embeddings.models import Base, RegulationChunk
from regulation_embeddings.pipeline import RegulationProcessor
from sentence_transformers import SentenceTransformer
from sqlalchemy import Column, DateTime, Integer, LargeBinary, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from tqdm import tqdm

# Set up logging similar to process_data.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/logs/embed_regulations.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def create_db(db_url: str):
    """Create the SQLite database and initialize tables."""
    logger.info("Creating/connecting to database")

    # Create database directory if it doesn't exist
    db_path = Path(db_url.replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    logger.info("Database tables created successfully")
    return engine


def extract_metadata(filename):
    """
    Extract title, chapter, and date from regulation filename.

    Args:
        filename (str): Name of the regulation file

    Returns:
        tuple: (title, chapter, date) as strings, or (None, None, None) if not found
    """
    title_chapter_match = re.search(r"title_(\d+)_chapter_([^_]+)", filename)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)

    title = title_chapter_match.group(1) if title_chapter_match else None
    chapter = title_chapter_match.group(2) if title_chapter_match else None
    date = date_match.group(1) if date_match else None

    return title, chapter, date


def chunk_regulation_xml(xml_path: str) -> List[Tuple[str, dict]]:
    """
    Split regulation XML into meaningful chunks based on the document structure.

    Args:
        xml_path (str): Path to the XML file

    Returns:
        list: List of tuples containing (chunk_text, metadata)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    chunks = []

    def clean_text(text: str) -> str:
        """Clean up text by removing extra whitespace and normalizing quotes."""
        if text is None:
            return ""
        return " ".join(text.split()).replace('"', '"').replace('"', '"')

    def process_section(section) -> dict:
        """Extract metadata from a section element."""
        metadata = {}
        # Get section number from HEAD
        head = section.find("HEAD")
        if head is not None:
            section_num = head.text.strip().replace("§", "").strip()
            metadata["section"] = section_num

        # Get hierarchy metadata if available
        hierarchy = section.get("hierarchy_metadata")
        if hierarchy:
            metadata["hierarchy"] = hierarchy

        # Extract additional metadata fields
        cross_references = section.findall(".//CROSSREF")
        if cross_references:
            metadata["cross_references"] = [
                clean_text(ref.text) for ref in cross_references if ref.text
            ]

        definitions = section.findall(".//DEF")
        if definitions:
            metadata["definitions"] = [
                clean_text(defn.text) for defn in definitions if defn.text
            ]

        enforcement_agencies = section.findall(".//ENFORCEMENT")
        if enforcement_agencies:
            metadata["enforcement_agencies"] = [
                clean_text(agency.text)
                for agency in enforcement_agencies
                if agency.text
            ]

        last_revision = section.find(".//LASTREV")
        if last_revision is not None:
            metadata["date_of_last_revision"] = clean_text(last_revision.text)

        intent = section.find(".//INTENT")
        if intent is not None:
            metadata["regulatory_intent"] = clean_text(intent.text)

        return metadata

    def extract_text_with_context(element) -> str:
        """Extract text from an element including its header for context."""
        texts = []
        head = element.find("HEAD")
        if head is not None:
            texts.append(head.text.strip())

        for child in element:
            if child.tag == "P":
                if child.text:
                    texts.append(clean_text(child.text))
            elif child.tag not in ["HEAD"]:  # Skip headers as they're already processed
                if child.text:
                    texts.append(clean_text(child.text))

        return " ".join(texts)

    # Process each major division (PART, SECTION, etc.)
    for div in root.findall(".//DIV8"):  # Find all section-level divisions
        section_metadata = process_section(div)

        # Create chunks based on paragraphs or subsections
        paragraphs = div.findall(".//P")
        if paragraphs:
            current_chunk = []
            current_length = 0
            max_chunk_length = 1000  # Adjust this value as needed

            for p in paragraphs:
                text = clean_text(p.text) if p.text else ""

                # If adding this paragraph would exceed max length, save current chunk
                if current_length + len(text) > max_chunk_length and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append((chunk_text, section_metadata))
                    current_chunk = []
                    current_length = 0

                current_chunk.append(text)
                current_length += len(text)

            # Add any remaining text as final chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append((chunk_text, section_metadata))
        else:
            # If no paragraphs found, create one chunk from the whole section
            text = extract_text_with_context(div)
            if text.strip():
                chunks.append((text, section_metadata))

    return chunks


def process_agencies():
    """
    Process all regulation files, create embeddings, and store in database.
    """
    # Initialize the embedding model
    logger.info("Loading embedding model")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Set up database
    engine = create_db()
    Session = sessionmaker(bind=engine)
    session = Session()

    data_dir = Path("data/agencies")
    logger.info(f"Reading from directory: {data_dir}")

    # Count total files
    total_files = sum(
        len(list(agency_dir.glob("xml/*.xml")))  # Changed to look for XML files
        for agency_dir in data_dir.iterdir()
        if agency_dir.is_dir()
    )

    processed_files = 0

    # Create progress bar for all files
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for agency_dir in data_dir.iterdir():
            if not agency_dir.is_dir():
                continue

            agency_name = agency_dir.name
            xml_dir = agency_dir / "xml"  # Changed to xml directory

            if not xml_dir.exists():
                logger.warning(f"No XML directory found for agency: {agency_name}")
                continue

            logger.info(f"Processing agency: {agency_name}")

            for xml_file in xml_dir.glob("*.xml"):
                try:
                    # Extract metadata
                    title, chapter, date = extract_metadata(xml_file.name)
                    if not all([title, chapter, date]):
                        logger.warning(
                            f"Could not extract metadata from: {xml_file.name}"
                        )
                        pbar.update(1)
                        continue

                    # Process XML and create chunks
                    chunks = chunk_regulation_xml(xml_file)

                    # Create embeddings and store chunks with progress bar
                    for chunk_index, (chunk_text, section_metadata) in tqdm(
                        enumerate(chunks),
                        desc=f"Processing chunks for {xml_file.name}",
                        leave=False,
                    ):
                        # Create embedding
                        embedding = model.encode(chunk_text)

                        # Generate embeddings for additional metadata fields
                        cross_references_embedding = model.encode(
                            " ".join(section_metadata.get("cross_references", []))
                        )
                        definitions_embedding = model.encode(
                            " ".join(section_metadata.get("definitions", []))
                        )
                        authority_embedding = model.encode(
                            " ".join(section_metadata.get("enforcement_agencies", []))
                        )

                        # Merge metadata embeddings with the main text embedding
                        enriched_embedding = np.concatenate(
                            [
                                embedding,
                                cross_references_embedding,
                                definitions_embedding,
                                authority_embedding,
                            ]
                        )

                        # Create database record
                        chunk_record = RegulationChunk(
                            agency=agency_name,
                            title=title,
                            chapter=chapter,
                            date=date,
                            chunk_text=chunk_text,
                            chunk_index=chunk_index,
                            embedding=enriched_embedding.tobytes(),
                            section=section_metadata.get("section"),
                            hierarchy=section_metadata.get("hierarchy"),
                        )

                        session.add(chunk_record)

                    processed_files += 1
                    logger.info(f"Processed {xml_file.name} into {len(chunks)} chunks")

                    # Commit after each file to avoid memory issues
                    session.commit()

                except Exception as e:
                    logger.error(f"Error processing file {xml_file}: {str(e)}")
                    session.rollback()

                finally:
                    pbar.update(1)

    logger.info(
        f"Processing complete. Processed {processed_files} of {total_files} files"
    )
    session.close()


def search_similar_chunks(query_text: str, db_url: str, n_results: int = 5):
    """
    Search for regulation chunks similar to the query text using cosine similarity
    and efficient batch processing.

    Args:
        query_text (str): The search query
        db_url (str): Database connection URL
        n_results (int): Number of results to return

    Returns:
        list: Top n_results similar chunks with metadata
    """
    # Load model and create embedding for query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query_text, normalize_embeddings=True)

    # Connect to database
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Process chunks in batches to avoid memory issues
        batch_size = 1000
        results = []

        # Get total count for progress tracking
        total_chunks = session.query(RegulationChunk).count()

        for offset in tqdm(range(0, total_chunks, batch_size), desc="Searching chunks"):
            # Get batch of chunks
            chunks = (
                session.query(RegulationChunk).offset(offset).limit(batch_size).all()
            )

            # Convert embeddings to numpy array
            chunk_embeddings = np.vstack(
                [np.frombuffer(chunk.embedding, dtype=np.float32) for chunk in chunks]
            )

            # Normalize embeddings for cosine similarity
            chunk_embeddings = (
                chunk_embeddings
                / np.linalg.norm(chunk_embeddings, axis=1)[:, np.newaxis]
            )

            # Calculate cosine similarities for the batch
            similarities = np.dot(chunk_embeddings, query_embedding)

            # Add results with metadata
            batch_results = [
                (
                    chunks[i].chunk_text,
                    float(similarities[i]),
                    {
                        "agency": chunks[i].agency,
                        "title": chunks[i].title,
                        "chapter": chunks[i].chapter,
                        "date": chunks[i].date,
                        "section": chunks[i].section,
                        "hierarchy": chunks[i].hierarchy,
                    },
                )
                for i in range(len(chunks))
            ]

            results.extend(batch_results)

            # Keep only top n_results so far to save memory
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:n_results]

        return results

    finally:
        session.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process and embed regulation documents."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yml"),
        help="Path to configuration file",
    )
    return parser.parse_args()


def main():
    """Run the regulation embedding pipeline."""
    parser = argparse.ArgumentParser(
        description="Process and embed regulation documents."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yml"),
        help="Path to configuration file",
    )
    args = parser.parse_args()

    try:
        # Load configuration
        print("Loading configuration...")
        config = Config.from_yaml(args.config)

        # Create database and tables
        create_db(config.database.db_url)

        # Initialize and run processor
        print("\nInitializing processor...")
        processor = RegulationProcessor(config)

        print("\nStarting processing...")
        processor.process_directory(config.processing.data_dir)

        print("\nProcessing complete!")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise
    finally:
        print("\nDone")


if __name__ == "__main__":
    # Comment out or remove these lines
    # config = Config.from_yaml(Path("config/default.yml"))
    # query = "What are the three main types of research misconduct according to USDA regulations?"
    # results = search_similar_chunks(query, config.database.db_url)

    # Instead, run the main function to process and embed regulations
    main()
