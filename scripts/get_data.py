"""
eCFR Data Downloader

This module provides functionality to download and manage federal regulations from the 
Electronic Code of Federal Regulations (eCFR) API. It handles downloading regulations 
for specific agencies, managing content versions, and storing both XML and plain text formats.

Example usage:
    # Basic usage to download all agencies' regulations
    from pathlib import Path
    from get_data import ECFRDownloader

    # Initialize downloader (defaults to ./data directory)
    downloader = ECFRDownloader()

    # Download all agencies and their regulations
    downloader.download_all_agencies()

    # Or download specific agency regulations
    regulations_map = downloader.generate_agency_regulations_map()
    downloader.download_agency_regulations(
        agency_slug="agriculture-department",
        regulations_map=regulations_map,
        date="2024-01-01"
    )

The downloaded content is organized in the following structure:
    data/
    ├── agencies.json                  # List of all agencies
    ├── agency_regulations_map.json    # Mapping of agencies to their regulations
    ├── download.log                   # Operation logs
    └── agencies/                      # Downloaded regulations by agency
        └── {agency-slug}/
            ├── xml/                   # Raw XML content
            ├── text/                  # Extracted plain text
            └── hashes/               # Content hashes for deduplication
"""

import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class ECFRDownloader:
    """
    A downloader for agencies and regulations from the eCFR API.

    Downloads and manages XML and text content of federal regulations from the Electronic
    Code of Federal Regulations (eCFR) API. Handles content versioning and deduplication
    using content hashes.
    """

    AGENCIES_URL = "https://www.ecfr.gov/api/admin/v1/agencies.json"
    TITLE_URL_TEMPLATE = (
        "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
    )
    VERSIONS_URL_TEMPLATE = (
        "https://www.ecfr.gov/api/versioner/v1/versions/title-{title}.json"
    )

    def __init__(
        self, base_dir: Path = Path("data"), user_agent: str = "ECFRDataCollector/1.0"
    ):
        """
        Initialize the downloader with a base directory and a persistent requests session.
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept": "application/json"}
        )

    def download_agencies(self) -> Optional[Dict[str, Any]]:
        """
        Download the list of agencies from the eCFR API and save it to agencies.json.
        """
        try:
            response = self.session.get(self.AGENCIES_URL)
            response.raise_for_status()
            data = response.json()

            output_file = self.base_dir / "agencies.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logging.info("Downloaded agencies data to %s", output_file)
            return data

        except requests.RequestException as e:
            logging.error("Error downloading agencies data: %s", e)
            return None

    def find_latest_available_date(self) -> str:
        """
        Find the most recent date with available content by checking for a valid XML resource.
        """
        current_date = datetime.now()
        for i in range(30):
            test_date = (current_date - timedelta(days=i)).strftime("%Y-%m-%d")
            url = self.TITLE_URL_TEMPLATE.format(
                date=test_date, title=1
            )  # Title is arbitrary here
            try:
                response = self.session.head(url)
                if response.status_code == 200:
                    logging.info("Found latest available date: %s", test_date)
                    return test_date
            except requests.RequestException:
                continue

        fallback_date = "2024-01-01"
        logging.warning("Could not find recent date, using fallback: %s", fallback_date)
        return fallback_date

    @staticmethod
    def extract_text_from_xml(
        xml_content: str, chapter: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract and clean raw text from XML content, optionally filtering by chapter.

        Args:
            xml_content: The XML content string to parse
            chapter: Optional chapter number within the title

        Returns:
            str: Cleaned plain text content, or None if parsing fails
        """
        try:
            root = ET.fromstring(xml_content)

            # First find the correct chapter - handle both XML formats
            chapter_root = None

            # Try to find chapter in full title format first
            if chapter and root.tag == "ECFR":
                # Navigate through the hierarchy to find Chapter III
                chapter_elem = root.find(f".//DIV3[@N='{chapter}'][@TYPE='CHAPTER']")
                if chapter_elem is not None:
                    chapter_root = chapter_elem
                    logging.info(f"Found chapter {chapter} in full title XML")

            # If we didn't find a chapter and the root is already a DIV3, use it
            if chapter_root is None and root.tag == "DIV3" and root.get("N") == chapter:
                chapter_root = root
                logging.info(f"XML already at chapter {chapter} level")

            if chapter_root is None:
                logging.error(f"Could not find chapter {chapter} in XML structure")
                return None

            # Now extract text from the chapter
            text_parts = []
            for elem in chapter_root.iter():
                # Skip certain elements that might contain non-regulation text
                if elem.tag in ["PRTPAGE", "FTREF", "CITA", "SOURCE", "HED", "AMDDATE"]:
                    continue

                # Special handling for section headers
                if elem.tag == "HEAD":
                    if elem.text and elem.text.strip():
                        text_parts.append("\n" + elem.text.strip() + "\n")
                # Regular text content
                elif elem.text and elem.text.strip():
                    text_parts.append(elem.text.strip())
                if elem.tail and elem.tail.strip():
                    text_parts.append(elem.tail.strip())

            if not text_parts:
                logging.error("No text content found in XML")
                return None

            text = "\n".join(text_parts)
            text = re.sub(r"\n\s*\n", "\n\n", text)  # Normalize line breaks
            text = re.sub(r"\s+", " ", text)  # Normalize whitespace

            if len(text.strip()) < 100:  # Arbitrary minimum length check
                logging.error(
                    f"Extracted text suspiciously short ({len(text.strip())} chars)"
                )
                return None

            return text.strip()

        except ET.ParseError as e:
            logging.error(f"XML parsing error: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error extracting text: {e}")
            return None

    def extract_metadata(self, section) -> Dict[str, Any]:
        """
        Extract enriched metadata from a section.

        Args:
            section: The XML section element to extract metadata from

        Returns:
            dict: Dictionary containing extracted metadata fields
        """
        metadata = {}

        # Extract section number and title
        head = section.find("HEAD")
        if head is not None:
            metadata["title"] = self.clean_text(head.text)
            if '§' in metadata:
                metadata["section"] = metadata["title"].split('§')[1].strip()

        # Extract authority (Legal basis)
        auth = section.find(".//AUTH")
        if auth is not None:
            metadata["authority"] = self.clean_text(auth.text)

        # Extract source of the regulation
        source = section.find(".//SOURCE")
        if source is not None:
            metadata["source"] = self.clean_text(source.text)

        # Extract cross-references to other sections
        cross_refs = section.findall(".//CROSSREF/P")
        metadata["cross_references"] = [self.clean_text(ref.text) for ref in cross_refs if ref.text]

        # Extract definitions of terms (if any)
        definitions = section.findall(".//P")
        metadata["definitions"] = [self.clean_text(d.text) for d in definitions if "<I>" in ET.tostring(d).decode()]

        # Extract last modification date
        last_update = section.find(".//CITA")
        if last_update is not None:
            metadata["last_revision"] = self.clean_text(last_update.text)

        # Extract enforcement agencies
        enforcement_agencies = section.findall(".//ENFORCEMENT")
        metadata["enforcement_agencies"] = [self.clean_text(agency.text) for agency in enforcement_agencies if agency.text]

        # Extract regulatory intent/purpose
        intent = section.find(".//INTENT")
        if intent is not None:
            metadata["regulatory_intent"] = self.clean_text(intent.text)

        return metadata

    def clean_text(self, text: str) -> str:
        """
        Normalize whitespace and remove unnecessary characters from text.

        Args:
            text: The text to clean

        Returns:
            str: Cleaned text
        """
        return " ".join(text.split()) if text else ""

    def extract_chunks_with_metadata(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        Extract all sections with enhanced metadata from XML content.

        Args:
            xml_content: The XML content string to parse

        Returns:
            list: List of dictionaries containing text chunks and metadata
        """
        try:
            root = ET.fromstring(xml_content)
            sections = root.xpath(".//DIV8")  # Adjust if needed
            extracted_data = []

            for section in sections:
                metadata = self.extract_metadata(section)
                paragraphs = [self.clean_text(p.text) for p in section.xpath(".//P") if p.text]

                if paragraphs:
                    chunk_text = " ".join(paragraphs)
                    extracted_data.append({"text": chunk_text, "metadata": metadata})

            return extracted_data

        except ET.ParseError as e:
            logging.error(f"XML parsing error: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error extracting chunks with metadata: {e}")
            return []

    def get_available_versions(
        self, title: str, chapter: Optional[str] = None, date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve available version for a given title and date.
        """
        url = self.TITLE_URL_TEMPLATE.format(date=date, title=title)
        if chapter:
            url += f"?chapter={chapter}"  # Add chapter parameter to URL

        max_retries = 3  # Increased from 1 to 3
        retry_delay = 3  # seconds
        timeout = 60  # Increased from 30 to 60 seconds

        for attempt in range(max_retries):
            try:
                # Do a GET request and store the content for later use
                response = self.session.get(url, timeout=timeout)

                if response.status_code == 200:
                    self.session.last_content = response.text
                    return [{"issue_date": date}]
                elif response.status_code == 404:
                    logging.info(f"No version available for Title {title} on {date}")
                    return []
                else:
                    response.raise_for_status()

            except requests.Timeout:
                if attempt < max_retries - 1:
                    logging.warning(
                        f"Timeout checking Title {title} for {date}. "
                        f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                logging.error(
                    f"Timeout checking Title {title} for {date} "
                    f"after {max_retries} attempts"
                )
                return []

            except requests.RequestException as e:
                logging.error(
                    "Error checking version for Title %s date %s: %s", title, date, e
                )
                return []

        return []

    def download_regulation(
        self, title: str, chapter: Optional[str], date: str, output_dir: Path
    ) -> bool:
        """
        Download XML content for a specific version of a regulation.

        Args:
            title: The CFR title number
            chapter: Optional chapter number within the title
            date: The target date for the regulation version (must be exact match)
            output_dir: Directory to store downloaded content

        Returns:
            bool: True if download successful or file already exists, False on failure
        """
        # Prepare directories for storing XML and text versions
        xml_dir = output_dir / "xml"
        text_dir = output_dir / "text"
        xml_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)

        # Check if text file already exists
        text_filename = f"title_{title}_chapter_{chapter}_{date}.txt"
        if (text_dir / text_filename).exists():
            logging.info(
                f"Text file already exists for Title {title} Chapter {chapter} date {date} - skipping download"
            )
            return True

        versions = self.get_available_versions(title, chapter, date)
        if not versions:
            logging.info(
                "No version found for Title %s Chapter %s on exact date %s",
                title,
                chapter,
                date,
            )
            return False

        version = versions[0]
        version_date = version["issue_date"]
        xml_filename = f"title_{title}_chapter_{chapter}_{version_date}.xml"
        text_filename = f"title_{title}_chapter_{chapter}_{version_date}.txt"

        try:
            # Use the content we already downloaded in get_available_versions
            if hasattr(self.session, "last_content"):
                xml_content = self.session.last_content
                delattr(self.session, "last_content")  # Clean up after using
            else:
                # Fallback in case content wasn't cached
                url = self.TITLE_URL_TEMPLATE.format(date=version_date, title=title)
                params = {"chapter": chapter} if chapter else {}
                logging.info(
                    f"Downloading Title {title} Chapter {chapter} version {version_date}..."
                )
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                xml_content = response.text

            # Parse the XML to get just the chapter content when needed
            try:
                root = ET.fromstring(xml_content)
                chapter_elem = None

                # If this is a full title XML, extract just the chapter
                if root.tag == "ECFR":
                    chapter_elem = root.find(
                        f".//DIV3[@N='{chapter}'][@TYPE='CHAPTER']"
                    )
                    if chapter_elem is not None:
                        # Convert chapter element back to string, preserving XML declaration
                        xml_content = '<?xml version="1.0"?>\n' + ET.tostring(
                            chapter_elem, encoding="unicode"
                        )
                        logging.info(f"Extracted chapter {chapter} XML from full title")

                # Save XML file (either full chapter or extracted chapter)
                with (xml_dir / xml_filename).open("w", encoding="utf-8") as f:
                    f.write(xml_content)

                # Extract and save the text version
                text_content = self.extract_text_from_xml(xml_content, chapter)
                if text_content:
                    try:
                        with (text_dir / text_filename).open(
                            "w", encoding="utf-8"
                        ) as f:
                            f.write(text_content)
                        logging.info(
                            f"Successfully extracted and saved text to {text_filename}"
                        )
                    except Exception as e:
                        logging.error(f"Failed to save text file {text_filename}: {e}")
                else:
                    logging.error(f"Failed to extract text from {xml_filename}")

                logging.info("Saved version %s to %s", version_date, output_dir)
                return True

            except ET.ParseError as e:
                logging.error(f"Error parsing XML content: {e}")
                return False

        except requests.RequestException as e:
            logging.error(
                "Error downloading Title %s Chapter %s version %s: %s",
                title,
                chapter,
                version_date,
                e,
            )
            return False

    def generate_agency_regulations_map(self) -> Optional[Dict[str, Any]]:
        """
        Generate and save a mapping of agencies to their regulations based on agencies.json data.
        """
        agencies_file = self.base_dir / "agencies.json"
        if not agencies_file.exists():
            logging.info("agencies.json not found - downloading...")
            agencies_data = self.download_agencies()
            if agencies_data is None:
                return None
        else:
            with agencies_file.open("r", encoding="utf-8") as f:
                agencies_data = json.load(f)

        regulations_map: Dict[str, Any] = {}
        for agency in agencies_data.get("agencies", []):
            agency_info = {
                "name": agency.get("name"),
                "short_name": agency.get("short_name"),
                "slug": agency.get("slug"),
                "regulations": [],
            }
            # Main agency regulations
            for ref in agency.get("cfr_references", []):
                regulation = {"title": ref.get("title"), "chapter": ref.get("chapter")}
                agency_info["regulations"].append(regulation)
            # Child agency regulations
            for child in agency.get("children", []):
                for ref in child.get("cfr_references", []):
                    regulation = {
                        "title": ref.get("title"),
                        "chapter": ref.get("chapter"),
                        "from_child": child.get("name"),
                    }
                    agency_info["regulations"].append(regulation)
            regulations_map[agency["slug"]] = agency_info

        map_file = self.base_dir / "agency_regulations_map.json"
        with map_file.open("w", encoding="utf-8") as f:
            json.dump(regulations_map, f, indent=2, ensure_ascii=False)

        logging.info("Generated regulations map saved to %s", map_file)
        return regulations_map

    def download_agency_regulations(
        self, agency_slug: str, regulations_map: Dict[str, Any], date: str
    ) -> None:
        """
        Download all regulations for a specific agency based on the provided regulations map.
        """
        agency_info = regulations_map.get(agency_slug)
        if not agency_info:
            logging.warning("Could not find agency with slug: %s", agency_slug)
            return

        agency_dir = self.base_dir / "agencies" / agency_slug
        agency_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Downloading regulations for %s...", agency_info.get("name"))
        for reg in agency_info.get("regulations", []):
            title = reg.get("title")
            chapter = reg.get("chapter")
            self.download_regulation(title, chapter, date, agency_dir)
            if reg.get("from_child"):
                logging.info("  (From %s)", reg.get("from_child"))

    def download_all_agencies(self, progress=None, task_id=None) -> None:
        """
        Orchestrate the download of regulations for all agencies across multiple years.
        """
        start_time = datetime.now()
        logging.info("Starting download at %s", start_time)

        # Generate list of dates (first of each year from 2017 to 2025)
        dates = [f"{year}-01-01" for year in range(2017, 2026)]
        logging.info("Will attempt to download versions for dates: %s", dates)

        # Ensure agencies data is available
        agencies_file = self.base_dir / "agencies.json"
        if not agencies_file.exists():
            logging.info("agencies.json not found - downloading...")
            agencies_data = self.download_agencies()
            if agencies_data is None:
                logging.error("Failed to download agencies data")
                return
        else:
            with agencies_file.open("r", encoding="utf-8") as f:
                agencies_data = json.load(f)

        # Load or generate the agency regulations map
        map_file = self.base_dir / "agency_regulations_map.json"
        if not map_file.exists():
            logging.info("agency_regulations_map.json not found - generating map...")
            regulations_map = self.generate_agency_regulations_map()
            if not regulations_map:
                logging.error("Failed to generate regulations map")
                return
        else:
            with map_file.open("r", encoding="utf-8") as f:
                regulations_map = json.load(f)

        total_agencies = len(regulations_map)
        total_operations = total_agencies * len(dates)
        completed_operations = 0

        logging.info("Downloading regulations for %d agencies...", total_agencies)

        for agency_slug in regulations_map.keys():
            logging.info("Processing agency: %s", agency_slug)
            for date in dates:
                logging.info(f"Attempting download for date: {date}")
                self.download_agency_regulations(agency_slug, regulations_map, date)
                completed_operations += 1
                if progress and task_id:
                    # Update progress based on total operations
                    progress.update(
                        task_id,
                        completed=(completed_operations * 100) / total_operations,
                    )

        end_time = datetime.now()
        duration = end_time - start_time
        logging.info("Finished download at %s", end_time)
        logging.info("Total execution time: %s", duration)


def __main__():
    # Configure logging to write to both file and console
    log_file = Path("data/logs/get_data.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    downloader = ECFRDownloader()
    downloader.download_all_agencies()
