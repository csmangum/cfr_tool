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
            chapter: Optional chapter number to filter content

        Returns:
            str: Cleaned plain text content, or None if parsing fails

        Extracts text from XML elements while preserving structure. Removes excess
        whitespace and normalizes line breaks.
        """
        try:
            root = ET.fromstring(xml_content)
            if chapter:
                # Look for the chapter DIV3 element
                chapter_elem = root.find(f".//DIV3[@N='{chapter}'][@TYPE='CHAPTER']")
                if chapter_elem is not None:
                    root = chapter_elem
                else:
                    logging.warning("Chapter %s not found in XML", chapter)
                    return None

            text_parts = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text_parts.append(elem.text.strip())
                if elem.tail and elem.tail.strip():
                    text_parts.append(elem.tail.strip())

            text = "\n".join(text_parts)
            text = re.sub(r"\n\s*\n", "\n\n", text)
            return text.strip()
        except ET.ParseError as e:
            logging.error("Error parsing XML: %s", e)
            return None

    def get_available_versions(
        self, title: str, chapter: Optional[str] = None, date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all available versions for a given title (and chapter/date filters), returning only the latest version.
        """
        url = self.VERSIONS_URL_TEMPLATE.format(title=title)
        params: Dict[str, Any] = {}
        if chapter:
            params["chapter"] = chapter
        if date:
            params["issue_date[lte]"] = date

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            versions_data = response.json()

            if versions_data.get("content_versions"):
                latest_version = max(
                    versions_data["content_versions"], key=lambda x: x["issue_date"]
                )
                return [latest_version]
            return []

        except requests.RequestException as e:
            logging.error(
                "Error getting versions for Title %s Chapter %s: %s", title, chapter, e
            )
            return []

    def download_regulation(
        self, title: str, chapter: Optional[str], date: str, output_dir: Path
    ) -> bool:
        """
        Download XML content for the most recent version of a regulation.

        Args:
            title: The CFR title number
            chapter: Optional chapter number within the title
            date: The target date for the regulation version
            output_dir: Directory to store downloaded content

        Returns:
            bool: True if download successful or content already exists, False on failure

        Downloads both XML and plain text versions. Uses content hashing to skip
        duplicate downloads. Stores content in xml/, text/, and hashes/ subdirectories.
        """
        versions = self.get_available_versions(title, chapter, date)
        if not versions:
            logging.info("No versions found for Title %s Chapter %s", title, chapter)
            return False

        # Prepare directories for storing hashes, XML, and text versions
        hash_dir = output_dir / "hashes"
        xml_dir = output_dir / "xml"
        text_dir = output_dir / "text"
        hash_dir.mkdir(exist_ok=True)
        xml_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)

        hash_file = hash_dir / f"title_{title}_chapter_{chapter}_hashes.json"
        try:
            with hash_file.open("r", encoding="utf-8") as f:
                content_hashes = json.load(f)
        except FileNotFoundError:
            content_hashes = {}

        version = versions[0]
        version_date = version["issue_date"]
        xml_filename = f"title_{title}_chapter_{chapter}_{version_date}.xml"
        text_filename = f"title_{title}_chapter_{chapter}_{version_date}.txt"

        # Build the URL for downloading the full XML content
        url = self.TITLE_URL_TEMPLATE.format(date=version_date, title=title)
        params = {"chapter": chapter} if chapter else {}

        try:
            logging.info(
                "Checking Title %s Chapter %s version %s...",
                title,
                chapter,
                version_date,
            )
            response = self.session.get(url, params=params)
            response.raise_for_status()
            xml_content = response.text
            content_hash = hashlib.sha256(xml_content.encode("utf-8")).hexdigest()

            # Skip if content hash already exists
            if content_hash in content_hashes.values():
                logging.info(
                    "Content for %s already exists (hash match) - skipping",
                    version_date,
                )
                return True

            # Save XML file
            with (xml_dir / xml_filename).open("w", encoding="utf-8") as f:
                f.write(xml_content)

            # Extract and save the text version
            text_content = self.extract_text_from_xml(xml_content, chapter)
            if text_content:
                with (text_dir / text_filename).open("w", encoding="utf-8") as f:
                    f.write(text_content)

            # Update stored hash values
            content_hashes[version_date] = content_hash
            with hash_file.open("w", encoding="utf-8") as f:
                json.dump(content_hashes, f, indent=2)

            logging.info("Saved new version %s to %s", version_date, output_dir)
            return True

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

    def download_all_agencies(self) -> None:
        """
        Orchestrate the download of regulations for all agencies.
        """
        start_time = datetime.now()
        logging.info("Starting download at %s", start_time)

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

        # Determine the date to use for all downloads
        date = self.find_latest_available_date()
        logging.info("Using date: %s", date)

        total_agencies = len(regulations_map)
        logging.info("Downloading regulations for %d agencies...", total_agencies)
        for i, agency_slug in enumerate(regulations_map.keys(), start=1):
            logging.info(
                "Processing agency %d of %d: %s", i, total_agencies, agency_slug
            )
            self.download_agency_regulations(agency_slug, regulations_map, date)

        end_time = datetime.now()
        duration = end_time - start_time
        logging.info("Finished download at %s", end_time)
        logging.info("Total execution time: %s", duration)


if __name__ == "__main__":
    # Configure logging to write to both file and console
    log_file = Path("data") / "data/logs/get_data.log"
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
