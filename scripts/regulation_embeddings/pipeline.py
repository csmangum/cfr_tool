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
            skipped_files.append(
                (xml_file, {"title": title, "chapter": chapter, "date": date})
            )
            continue

        agency = xml_file.parent.parent.name
        key = (agency, title, chapter)

        # Keep track of the latest date for each regulation
        if key not in latest_files or date > latest_files[key][0]:
            latest_files[key] = (date, xml_file)

    # Log debug information
    logger = logging.getLogger(__name__)
    logger.info(f"Found {total_xml_files} total XML files")
    logger.info(
        f"Skipped {len(skipped_files)} files due to metadata extraction failures:"
    )
    for file, meta in skipped_files:
        logger.info(f"  {file.name}: {meta}")
    logger.info(f"Selected {len(latest_files)} files for processing:")
    for key, (date, file) in latest_files.items():
        logger.info(f"  {key[0]} - Title {key[1]} Chapter {key[2]} ({date})")

    # Return only the file paths of the latest versions
    return [file_path for _, file_path in latest_files.values()]


class ProcessingError(Exception):
    """Base exception for regulation processing errors."""

    pass


class ChunkingError(ProcessingError):
    """Error during document chunking."""

    pass


class EmbeddingError(ProcessingError):
    """Error during embedding generation."""

    pass


class StorageError(ProcessingError):
    """Error during data storage."""

    pass


def process_single_file(
    file_path: Path, chunker: XMLChunker, logger: logging.Logger
) -> Tuple[List[Tuple[str, Dict]], Dict[str, str]]:
    """Process a single file with improved error handling."""
    try:
        # Extract metadata first to fail fast if invalid
        agency = file_path.parent.parent.name
        title, chapter, date = extract_metadata(file_path.name)

        if not all([title, chapter, date]):
            raise ChunkingError(
                f"Failed to extract metadata from {file_path.name}. "
                f"Got: title={title}, chapter={chapter}, date={date}"
            )

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

        logger.info(f"Generated {len(chunks)} chunks from {file_path.name}")
        return chunks, metadata

    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {str(e)}", exc_info=True)
        raise ChunkingError(f"Failed to process {file_path}: {str(e)}") from e


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

        # Add cleanup flag
        self._is_shutdown = False

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

        return FaissStore(dimension=384)  # dimension for all-MiniLM-L6-v2 model

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        """Clean up resources."""
        if not self._is_shutdown:
            self.logger.info("Shutting down RegulationProcessor...")
            self.embedder.close()  # Add close method to embedder if needed
            self.db_manager.close()
            self._is_shutdown = True

    def process_files_parallel(self, files: List[Path]) -> None:
        """Process multiple files in parallel using process pool.

        Args:
            files (List[Path]): List of XML file paths to process
        """
        if self._is_shutdown:
            raise RuntimeError("RegulationProcessor has been shut down")

        chunk_results = []
        failed_files = []

        # Process files in smaller batches
        batch_size = min(50, len(files))  # Adjust batch size based on file count
        num_batches = (len(files) + batch_size - 1) // batch_size

        self.logger.info(
            f"Processing {len(files)} files in {num_batches} batches "
            f"of up to {batch_size} files each"
        )

        try:
            for i in range(0, len(files), batch_size):
                batch_files = files[i : i + batch_size]
                self._process_batch(
                    batch_files, chunk_results, failed_files, i // batch_size + 1
                )

        except KeyboardInterrupt:
            self.logger.warning("Processing interrupted by user")
            raise

        except Exception as e:
            self.logger.error(f"Fatal error during processing: {str(e)}", exc_info=True)
            raise

        finally:
            # Log summary
            self.logger.info(
                f"Processing complete. Successfully processed "
                f"{len(files) - len(failed_files)} files"
            )
            if failed_files:
                self.logger.warning(f"Failed to process {len(failed_files)} files:")
                for file in failed_files:
                    self.logger.warning(f"  {file}")

    def _process_batch(
        self,
        batch_files: List[Path],
        chunk_results: List[Tuple],
        failed_files: List[Path],
        batch_num: int,
    ) -> None:
        """Process a batch of files with improved memory management."""
        self.logger.info(f"Processing batch {batch_num} ({len(batch_files)} files)")

        # Use a generator to process files to reduce memory usage
        def process_files():
            with ProcessPoolExecutor(
                max_workers=self.config.processing.max_workers
            ) as executor:
                process_file = functools.partial(
                    process_single_file, chunker=self.chunker, logger=self.logger
                )
                yield from executor.map(process_file, batch_files)

        # Process results as they are generated
        for file_path, result in zip(batch_files, process_files()):
            try:
                if result is None:
                    failed_files.append(file_path)
                    continue

                chunks, metadata = result
                if chunks:
                    # Process chunks immediately to free memory
                    self._process_chunk_results([(chunks, metadata)])
                else:
                    failed_files.append(file_path)

            except ProcessingError as e:
                self.logger.error(f"Failed to process {file_path}: {str(e)}")
                failed_files.append(file_path)
                continue

    def _process_chunk_results(
        self, chunk_results: List[Tuple[List[Tuple[str, Dict]], Dict[str, str]]]
    ) -> None:
        """Process and store the results of a batch of chunking."""
        chunk_groups = [chunks for chunks, _ in chunk_results]
        metadata_groups = [metadata for _, metadata in chunk_results]

        # Step 2: Batch embedding
        self.logger.info(f"Generating embeddings for {len(chunk_groups)} files")
        embeddings_groups = self._batch_embed_chunks(chunk_groups)

        # Step 3: Batch storage
        self.logger.info(f"Storing chunks and embeddings")
        self._batch_store_chunks(chunk_groups, embeddings_groups, metadata_groups)

    def _batch_embed_chunks(
        self, chunk_groups: List[List[Tuple[str, Dict]]]
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple groups of chunks in batches."""
        all_embeddings = []

        for chunks in tqdm(chunk_groups, desc="Generating embeddings"):
            try:
                # Process each chunk with its metadata
                batch_embeddings = []
                for chunk_text, metadata in chunks:
                    embedding = self.embedder.embed_text_with_metadata(
                        text=chunk_text,
                        metadata=metadata,
                        enrich=False,  #! TODO: Change to True when ready
                    )
                    batch_embeddings.append(embedding)

                all_embeddings.append(np.vstack(batch_embeddings))

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

        self.logger.info(
            f"Starting to store {total_chunks} chunks from {len(chunk_groups)} files"
        )

        for chunks, embeddings, metadata in tqdm(
            zip(chunk_groups, embeddings_groups, metadata_groups),
            desc="Storing chunks",
            total=len(chunk_groups),
        ):
            try:
                self.logger.info(
                    f"Storing {len(chunks)} chunks for agency {metadata['agency']}"
                )

                # Only store in SQL database for now
                self.db_manager.store_chunks_sql_only(chunks, embeddings, metadata)

                stored_chunks += len(chunks)
                self.logger.info(f"Successfully stored chunks for {metadata['agency']}")

            except Exception as e:
                failed_chunks += len(chunks)
                self.logger.error(
                    f"Error storing chunks for {metadata['agency']}: {str(e)}",
                    exc_info=True,
                )
                continue

        self.logger.info(
            f"Storage complete: {stored_chunks} chunks stored, {failed_chunks} chunks failed"
        )

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

    def process_file(self, file_path: Path) -> None:
        """Process a single regulation file.

        Args:
            file_path (Path): Path to the XML file to process

        Raises:
            Exception: If there is an error processing the file
        """
        # Convert to Path if string is passed
        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        self.logger.info(f"Processing single file: {file_path}")

        try:
            # Step 1: Generate chunks
            chunks, metadata = process_single_file(file_path, self.chunker)

            if not chunks:
                self.logger.warning(f"No chunks generated for {file_path}")
                return

            self.logger.info(f"Generated {len(chunks)} chunks")

            # Step 2: Generate embeddings
            embeddings = self._batch_embed_chunks([chunks])[0]
            self.logger.info(f"Generated embeddings of shape {embeddings.shape}")

            # Step 3: Store chunks and embeddings
            self._batch_store_chunks([chunks], [embeddings], [metadata])
            self.logger.info(f"Successfully stored chunks and embeddings")

        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {str(e)}", exc_info=True)
            raise
