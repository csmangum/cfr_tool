"""Module for handling document chunking strategies."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from lxml import etree
from lxml.etree import ParseError, XMLSyntaxError


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
    """Chunks XML documents based on their structure."""

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
    ):
        self.max_chunk_length = max_chunk_length
        self.min_chunk_length = min_chunk_length
        self.xml_tag_depth = xml_tag_depth
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        """Clean up text by removing extra whitespace and normalizing quotes."""
        if text is None:
            return ""
        import re

        # Remove extra whitespace including newlines
        text = re.sub(r"\s+", " ", text.strip())
        # Replace smart quotes and dashes with standard versions
        replacements = {
            '"': '"',
            '"': '"',
            """: "'", """: "'",
            "—": "-",
            "–": "-",
            "…": "...",
            "\u200b": "",  # zero-width space
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _extract_text_list(self, elements: List[etree._Element]) -> List[str]:
        """Helper function to extract and clean text from a list of elements."""
        return [self.clean_text(elem.text) for elem in elements if elem.text]

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
        # Try to parse existing hierarchy metadata
        try:
            hierarchy = element.get("hierarchy_metadata")
            if hierarchy:
                return json.loads(hierarchy)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse hierarchy metadata: {str(e)}")

        # Fallback: Build hierarchy from element structure
        return {
            label.lower(): parent[0].get("N")
            for tag, label in self.DIVISION_TAGS.items()
            if (parent := element.xpath(f"./ancestor-or-self::{tag}[@N]"))
        }

    def extract_section_metadata(self, element: etree._Element) -> Dict:
        """Extract metadata from a section element."""
        metadata = {"hierarchy": self.extract_hierarchy_metadata(element)}

        # Extract section number and title
        head = element.find("HEAD")
        if head is not None and head.text:
            head_text = head.text.strip()
            if "§" in head_text:
                section_num = head_text.split("§")[1].strip()
                metadata["section"] = section_num
            metadata["title"] = head_text

        # Find the parent DIV5 element
        div5 = element.xpath("./ancestor::DIV5[1]")
        if div5:
            div5 = div5[0]
            # Extract authority and source using helper function
            for field in ["AUTH", "SOURCE"]:
                elem = div5.find(f".//{field}")
                if elem is not None:
                    elem_copy = etree.fromstring(etree.tostring(elem))
                    for xref in elem_copy.findall(".//XREF"):
                        xref.getparent().remove(xref)
                    text = " ".join(elem_copy.xpath(".//text()")).strip()
                    metadata[field.lower()] = self.clean_text(text)

        # Extract lists using helper function
        for field, xpath in [
            ("cross_references", ".//CROSSREF"),
            ("definitions", ".//DEF"),
            ("enforcement_agencies", ".//ENFORCEMENT"),
        ]:
            elements = element.xpath(xpath)
            if elements:
                metadata[field] = self._extract_text_list(elements)

        # Extract single value fields
        for field, xpath in [
            ("date_of_last_revision", ".//LASTREV"),
            ("regulatory_intent", ".//INTENT"),
        ]:
            elem = element.find(xpath)
            if elem is not None:
                metadata[field] = self.clean_text(elem.text)

        return metadata

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
                    for chunk_text, chunk_metadata in self.iter_chunks(div, metadata):
                        chunks.append((chunk_text, chunk_metadata))

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
