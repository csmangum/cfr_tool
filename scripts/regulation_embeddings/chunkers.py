"""Module for handling document chunking strategies."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from lxml import etree
from lxml.etree import ParseError, XMLSyntaxError
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.tokenize import sent_tokenize
from texttiling import TextTilingTokenizer


class XMLParsingError(Exception):
    """Custom exception for XML parsing errors."""

    pass


class ChunkingError(Exception):
    """Custom exception for chunking errors."""

    pass


class BaseChunker(ABC):
    """Abstract base class for document chunking strategies."""

    @abstractmethod
    def chunk_document(self, document_path: Path) -> List[Tuple[str, Dict]]:
        """Chunk a document into segments with metadata."""
        pass


class XMLChunker(BaseChunker):
    """Chunks XML documents based on their structure and semantic similarity."""

    # XML namespaces commonly found in regulation documents
    NAMESPACES = {
        "cfr": "http://www.ecfr.gov/schema/2.0",
        "dc": "http://purl.org/dc/elements/1.1/",
    }

    # Tags that represent logical divisions in the document
    DIVISION_TAGS = {
        "DIV1": "Title",
        "DIV2": "Chapter",
        "DIV3": "Part",
        "DIV4": "Subpart",
        "DIV5": "Section",
        "DIV6": "Subsection",
        "DIV7": "Paragraph",
        "DIV8": "Subparagraph",
    }

    def __init__(
        self,
        max_chunk_length: int = 1000,
        xml_tag_depth: str = ".//DIV8",
        min_chunk_length: int = 100,
        similarity_threshold: float = 0.7,
        use_text_tiling: bool = False,
        use_bert_segmentation: bool = False,
    ):
        self.max_chunk_length = max_chunk_length
        self.min_chunk_length = min_chunk_length
        self.xml_tag_depth = xml_tag_depth
        self.similarity_threshold = similarity_threshold
        self.use_text_tiling = use_text_tiling
        self.use_bert_segmentation = use_bert_segmentation
        self.logger = logging.getLogger(__name__)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.text_tiling_tokenizer = TextTilingTokenizer()

    def clean_text(self, text: str) -> str:
        """Clean up text by removing extra whitespace and normalizing quotes."""
        if text is None:
            return ""
        # More thorough text cleaning
        text = " ".join(text.split())
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(""", "'").replace(""", "'")
        text = text.replace("—", "-").replace("–", "-")
        return text

    def parse_xml(self, document_path: Path) -> etree._Element:
        """
        Parse XML document with robust error handling.

        Args:
            document_path: Path to XML file

        Returns:
            lxml.etree._Element: Parsed XML tree

        Raises:
            XMLParsingError: If document cannot be parsed
        """
        try:
            # Use a custom parser with recovery mode
            parser = etree.XMLParser(recover=True, remove_blank_text=True)
            tree = etree.parse(str(document_path), parser)

            # Log any parser errors
            for error in parser.error_log:
                self.logger.warning(
                    f"XML parsing issue in {document_path.name}: {error.message}"
                )

            return tree.getroot()

        except (XMLSyntaxError, ParseError, OSError) as e:
            raise XMLParsingError(f"Failed to parse {document_path}: {str(e)}")

    def extract_hierarchy_metadata(self, element: etree._Element) -> Dict:
        """
        Extract and parse hierarchy metadata from element attributes.

        Args:
            element: XML element to extract metadata from

        Returns:
            dict: Parsed hierarchy metadata
        """
        try:
            hierarchy = element.get("hierarchy_metadata")
            if hierarchy:
                return json.loads(hierarchy)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse hierarchy metadata: {str(e)}")

        # Fallback: Build hierarchy from element structure
        hierarchy = {}
        for tag, label in self.DIVISION_TAGS.items():
            parent = element.xpath(f"./ancestor-or-self::{tag}[@N]")
            if parent:
                hierarchy[label.lower()] = parent[0].get("N")

        return hierarchy

    def extract_section_metadata(self, element: etree._Element) -> Dict:
        """Extract metadata from a section element."""
        metadata = {"hierarchy": self.extract_hierarchy_metadata(element)}

        # Extract cross-references
        cross_references = element.xpath(".//CROSSREF")
        if cross_references:
            metadata["cross_references"] = [self.clean_text(ref.text) for ref in cross_references if ref.text]

        # Extract definitions
        definitions = element.xpath(".//DEF")
        if definitions:
            metadata["definitions"] = [self.clean_text(defn.text) for defn in definitions if defn.text]

        # Extract enforcement agencies
        enforcement_agencies = element.xpath(".//ENFORCEMENT")
        if enforcement_agencies:
            metadata["enforcement_agencies"] = [self.clean_text(agency.text) for agency in enforcement_agencies if agency.text]

        # Extract section number and title
        head = element.find("HEAD")
        if head is not None:
            head_text = head.text.strip()
            # Parse section number (§ 301.1)
            if "§" in head_text:
                section_num = head_text.split("§")[1].strip()
                metadata["section"] = section_num

            metadata["title"] = head_text

        # Extract authority information
        auth = element.find(".//AUTH")
        if auth is not None:
            metadata["authority"] = self.clean_text(auth.text)

        # Extract source information
        source = element.find(".//SOURCE")
        if source is not None:
            metadata["source"] = self.clean_text(source.text)

        # Extract date of last revision
        last_revision = element.find(".//LASTREV")
        if last_revision is not None:
            metadata["date_of_last_revision"] = self.clean_text(last_revision.text)

        # Extract regulatory intent/purpose
        intent = element.find(".//INTENT")
        if intent is not None:
            metadata["regulatory_intent"] = self.clean_text(intent.text)

        return metadata

    def compute_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Compute sentence embeddings using Sentence Transformers."""
        return self.model.encode(sentences, convert_to_numpy=True)

    def identify_chunk_boundaries(self, sentences: List[str]) -> List[int]:
        """Identify chunk boundaries based on semantic similarity."""
        embeddings = self.compute_sentence_embeddings(sentences)
        similarities = np.array(
            [np.dot(embeddings[i], embeddings[i + 1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]))
             for i in range(len(embeddings) - 1)]
        )
        boundaries = [i + 1 for i, sim in enumerate(similarities) if sim < self.similarity_threshold]
        return boundaries

    def iter_chunks(
        self, element: etree._Element, metadata: Dict
    ) -> Iterator[Tuple[str, Dict]]:
        """
        Iterate through chunks in an element, respecting logical boundaries.

        Args:
            element: XML element to chunk
            metadata: Base metadata for the chunks

        Yields:
            Tuple[str, Dict]: Chunk text and its metadata
        """
        current_chunk = []
        current_length = 0

        # Process paragraphs while preserving structure
        for p in element.xpath(".//P"):
            # Get paragraph identifier if available
            p_id = p.get("N", "")
            text = self.clean_text(p.text) if p.text else ""

            # Add paragraph identifier to text if available
            if p_id:
                text = f"({p_id}) {text}"

            # Check if adding this paragraph would exceed max length
            if current_length + len(text) > self.max_chunk_length and current_chunk:
                # Yield current chunk if it meets minimum length
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_length:
                    yield chunk_text, metadata
                current_chunk = []
                current_length = 0

            current_chunk.append(text)
            current_length += len(text)

        # Yield any remaining text
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_length:
                yield chunk_text, metadata

    def chunk_document(self, document_path: Path) -> List[Tuple[str, Dict]]:
        """
        Chunk an XML document into segments with metadata.

        Args:
            document_path: Path to XML document

        Returns:
            List[Tuple[str, Dict]]: List of chunks with their metadata

        Raises:
            ChunkingError: If document cannot be chunked properly
        """
        try:
            root = self.parse_xml(document_path)
            chunks = []

            # Process each division at the specified depth
            for div in root.xpath(self.xml_tag_depth):
                try:
                    # Extract metadata for this division
                    metadata = self.extract_section_metadata(div)

                    # Generate chunks for this division
                    if self.use_text_tiling:
                        # Use TextTiling for chunking
                        text = " ".join([self.clean_text(p.text) for p in div.xpath(".//P") if p.text])
                        tiling_chunks = self.text_tiling_tokenizer.tokenize(text)
                        for chunk in tiling_chunks:
                            if len(chunk) >= self.min_chunk_length:
                                chunks.append((chunk, metadata))
                    elif self.use_bert_segmentation:
                        # Placeholder for BERT-based segmentation
                        pass
                    else:
                        # Use semantic similarity for chunking
                        sentences = [self.clean_text(p.text) for p in div.xpath(".//P") if p.text]
                        boundaries = self.identify_chunk_boundaries(sentences)
                        start = 0
                        for boundary in boundaries:
                            chunk_text = " ".join(sentences[start:boundary])
                            if len(chunk_text) >= self.min_chunk_length:
                                chunks.append((chunk_text, metadata))
                            start = boundary
                        # Add remaining sentences as the last chunk
                        chunk_text = " ".join(sentences[start:])
                        if len(chunk_text) >= self.min_chunk_length:
                            chunks.append((chunk_text, metadata))

                except Exception as e:
                    # Log error but continue processing other divisions
                    self.logger.error(
                        f"Error processing division in {document_path.name}: {str(e)}"
                    )

            if not chunks:
                self.logger.warning(f"No chunks generated for {document_path.name}")

            return chunks

        except XMLParsingError as e:
            raise ChunkingError(f"Failed to chunk document: {str(e)}")
