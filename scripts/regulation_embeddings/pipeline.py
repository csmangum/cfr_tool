"""Main pipeline for processing and embedding regulations."""

import functools
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .chunkers import XMLChunker
from .config import Config
from .embedders import SentenceTransformerEmbedder
from .storage import DatabaseManager
from .vector_store_base import VectorStore
from .vector_stores import FaissStore


def extract_metadata(filename: str) -> Tuple[str, str, str]:
    """Extract metadata from filename."""
    import re

    title_chapter_match = re.search(r"title_(\d+)_chapter_([^_]+)", filename)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)

    title = title_chapter_match.group(1) if title_chapter_match else None
    chapter = title_chapter_match.group(2) if title_chapter_match else None
    date = date_match.group(1) if date_match else None

    return title, chapter, date


def get_latest_regulation_files(data_dir: Path) -> List[Path]:
    """
    Get the most recent version of each regulation.
    """
    latest_files = {}  # (agency, title, chapter) -> (date, file_path)
    
    # Debug counter
    total_xml_files = 0
    skipped_files = []
    
    # Iterate through all XML files
    for xml_file in data_dir.rglob("*/xml/*.xml"):
        total_xml_files += 1
        title, chapter, date = extract_metadata(xml_file.name)
        
        if not all([title, chapter, date]):
            skipped_files.append((xml_file, {'title': title, 'chapter': chapter, 'date': date}))
            continue

        agency = xml_file.parent.parent.name
        key = (agency, title, chapter)

        # Keep track of the latest date for each regulation
        if key not in latest_files or date > latest_files[key][0]:
            latest_files[key] = (date, xml_file)

    # Log debug information
    logger = logging.getLogger(__name__)
    logger.info(f"Found {total_xml_files} total XML files")
    logger.info(f"Skipped {len(skipped_files)} files due to metadata extraction failures:")
    for file, meta in skipped_files:
        logger.info(f"  {file.name}: {meta}")
    logger.info(f"Selected {len(latest_files)} files for processing:")
    for key, (date, file) in latest_files.items():
        logger.info(f"  {key[0]} - Title {key[1]} Chapter {key[2]} ({date})")

    # Return only the file paths of the latest versions
    return [file_path for _, file_path in latest_files.values()]


def process_single_file(
    file_path: Path, chunker: XMLChunker
) -> Tuple[List[Tuple[str, Dict]], Dict[str, str]]:
    """Process a single file - separated for multiprocessing."""
    logger = logging.getLogger(__name__)
    try:
        agency = file_path.parent.parent.name
        title, chapter, date = extract_metadata(file_path.name)

        metadata = {"agency": agency, "title": title, "chapter": chapter, "date": date}
        
        logger.info(f"Processing {file_path.name} for agency {agency}")
        
        # Get chunks with detailed logging
        chunks = chunker.chunk_document(file_path)
        
        if not chunks:
            logger.warning(
                f"No chunks generated for {file_path.name}. "
                f"This could indicate an issue with the XML structure or empty content."
            )
            return [], metadata
            
        logger.info(f"Successfully generated {len(chunks)} chunks from {file_path.name}")
        return chunks, metadata

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
        raise


class RegulationProcessor:
    """Coordinates the processing, embedding, and storage of regulations."""

    def __init__(self, config: Config):
        self.config = config

        # Initialize components with configuration
        self.chunker = XMLChunker(
            max_chunk_length=config.chunker.max_chunk_length,
            xml_tag_depth=config.chunker.xml_tag_depth,
        )

        self.embedder = SentenceTransformerEmbedder(
            model_name=config.embedder.model_name,
            batch_size=config.embedder.batch_size,
            device=config.embedder.device,
        )

        # Initialize vector store if configured
        vector_store = self._setup_vector_store()

        self.db_manager = DatabaseManager(
            db_url=config.database.db_url,
            batch_size=config.database.batch_size,
            vector_store=vector_store,
        )

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Set up parallel processing
        self.max_workers = min(
            multiprocessing.cpu_count() - 1, 4
        )  # Reduced number of workers

    def _setup_logging(self):
        """Configure logging based on configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.config.processing.log_file),
                logging.StreamHandler(),
            ],
        )

    def _setup_vector_store(self) -> Optional[VectorStore]:
        """Set up FAISS vector store."""
        vector_store_config = self.config.vector_store

        return FaissStore(
            dimension=384  # dimension for all-MiniLM-L6-v2 model
        )

    def process_files_parallel(self, files: List[Path]) -> None:
        """Process multiple files in parallel using process pool."""
        chunk_results = []
        failed_files = []

        # Process files in smaller batches
        batch_size = 50  # Reduced batch size for better tracking
        for i in range(0, len(files), batch_size):
            batch_files = files[i : i + batch_size]
            
            self.logger.info(f"Processing batch {i//batch_size + 1} ({len(batch_files)} files)")

            # Step 1: Parallel chunking using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                process_file = functools.partial(process_single_file, chunker=self.chunker)
                
                # Process the batch
                futures = [executor.submit(process_file, file_path) for file_path in batch_files]
                
                # Use tqdm to show progress
                for file_path, future in zip(batch_files, tqdm(futures, desc=f"Chunking batch {i//batch_size + 1}")):
                    try:
                        chunks, metadata = future.result()
                        if chunks:  # Only add if we got valid chunks
                            chunk_results.append((chunks, metadata))
                        else:
                            failed_files.append(file_path)
                    except Exception as e:
                        self.logger.error(f"Failed to process {file_path}: {str(e)}")
                        failed_files.append(file_path)
                        continue

            # Process accumulated results for this batch
            if chunk_results:
                try:
                    # Prepare batches for embedding and storage
                    chunk_groups = [chunks for chunks, _ in chunk_results]
                    metadata_groups = [metadata for _, metadata in chunk_results]

                    # Step 2: Batch embedding
                    self.logger.info(f"Generating embeddings for {len(chunk_groups)} files")
                    embeddings_groups = self._batch_embed_chunks(chunk_groups)

                    # Step 3: Batch storage
                    self.logger.info(f"Storing chunks and embeddings")
                    self._batch_store_chunks(chunk_groups, embeddings_groups, metadata_groups)

                except Exception as e:
                    self.logger.error(f"Error processing batch: {str(e)}")

                # Clear results after processing to free memory
                chunk_results = []

        # Log summary
        self.logger.info(f"Processing complete. Successfully processed {len(files) - len(failed_files)} files")
        if failed_files:
            self.logger.warning(f"Failed to process {len(failed_files)} files:")
            for file in failed_files:
                self.logger.warning(f"  {file}")

    def _batch_embed_chunks(
        self, chunk_groups: List[List[Tuple[str, Dict]]]
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple groups of chunks in batches."""
        all_embeddings = []

        for chunks in tqdm(chunk_groups, desc="Generating embeddings"):
            try:
                chunk_texts = [chunk[0] for chunk in chunks]
                embeddings = self.embedder.embed_chunks(chunk_texts)
                all_embeddings.append(embeddings)
            except Exception as e:
                self.logger.error(f"Error generating embeddings: {str(e)}")
                raise

        return all_embeddings

    def _batch_store_chunks(
        self,
        chunk_groups: List[List[Tuple[str, Dict]]],
        embeddings_groups: List[np.ndarray],
        metadata_groups: List[Dict[str, str]],
    ) -> None:
        """Store multiple groups of chunks and their embeddings in batches."""
        total_chunks = sum(len(chunks) for chunks in chunk_groups)
        stored_chunks = 0
        failed_chunks = 0
        
        self.logger.info(f"Starting to store {total_chunks} chunks from {len(chunk_groups)} files")
        
        for chunks, embeddings, metadata in tqdm(
            zip(chunk_groups, embeddings_groups, metadata_groups), 
            desc="Storing chunks",
            total=len(chunk_groups)
        ):
            try:
                self.logger.info(f"Storing {len(chunks)} chunks for agency {metadata['agency']}")
                
                # Only store in SQL database for now
                self.db_manager.store_chunks_sql_only(chunks, embeddings, metadata)
                
                stored_chunks += len(chunks)
                self.logger.info(f"Successfully stored chunks for {metadata['agency']}")
                
            except Exception as e:
                failed_chunks += len(chunks)
                self.logger.error(
                    f"Error storing chunks for {metadata['agency']}: {str(e)}", 
                    exc_info=True
                )
                continue

        self.logger.info(f"Storage complete: {stored_chunks} chunks stored, {failed_chunks} chunks failed")

    def process_directory(self, data_dir: str) -> None:
        """Process regulation files from directory."""
        # Convert string to Path if needed
        data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

        # Get the latest version of each regulation
        xml_files = get_latest_regulation_files(data_dir)

        if not xml_files:
            self.logger.warning(f"No regulation files found in {data_dir}")
            return

        # Log which files will be processed
        self.logger.info(f"Found {len(xml_files)} regulation files to process")

        # Process files in parallel
        self.process_files_parallel(xml_files)
