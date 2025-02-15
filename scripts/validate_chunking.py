"""Script to validate XMLChunker's chunking logic and metadata extraction."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from lxml import etree

from regulation_embeddings.chunkers import XMLChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_metadata_extraction(xml_path: Path) -> None:
    """Validate metadata extraction from XML document."""
    
    # Parse XML directly to compare against chunker output
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    
    # Initialize chunker
    chunker = XMLChunker()
    chunks = chunker.chunk_document(xml_path)
    
    # Track statistics
    total_sections = len(root.xpath(".//DIV8"))
    div5_elements = root.xpath(".//DIV5")
    
    logger.info("\nDocument Statistics:")
    logger.info("-" * 50)
    logger.info(f"Total sections (DIV8): {total_sections}")
    logger.info(f"Total parts (DIV5): {len(div5_elements)}")
    logger.info(f"Total chunks generated: {len(chunks)}")
    
    # Validate a sample DIV5 and its sections
    sample_div5 = div5_elements[0]
    part_num = sample_div5.get('N')
    
    logger.info(f"\nValidating Part {part_num}:")
    logger.info("-" * 50)
    
    # Get expected authority and source, excluding XREF elements
    auth = sample_div5.find(".//AUTH")
    source = sample_div5.find(".//SOURCE")
    
    def get_clean_text(element):
        """Get text content excluding XREF elements."""
        if element is None:
            return None
        # Remove XREF elements before getting text
        for xref in element.findall(".//XREF"):
            xref.getparent().remove(xref)
        return " ".join(element.xpath(".//text()")).strip()
    
    expected_auth = get_clean_text(auth)
    expected_source = get_clean_text(source)
    
    logger.info("Expected Authority:")
    logger.info(expected_auth[:200] + "..." if expected_auth else "None")
    logger.info("\nExpected Source:")
    logger.info(expected_source[:200] + "..." if expected_source else "None")
    
    # Get sections from this part
    sections = sample_div5.xpath(".//DIV8")
    section_nums = [section.xpath(".//HEAD/text()")[0].split("§")[1].strip() 
                   if "§" in section.xpath(".//HEAD/text()")[0]
                   else section.xpath(".//HEAD/text()")[0].strip()
                   for section in sections]
    
    logger.info(f"\nFound {len(sections)} sections in Part {part_num}:")
    logger.info(", ".join(section_nums))
    
    # Check chunks for these sections
    matching_chunks = [(chunk, meta) for chunk, meta in chunks 
                      if meta.get('section') in section_nums]
    
    logger.info(f"\nFound {len(matching_chunks)} chunks for these sections")
    
    # Validate metadata for first chunk
    if matching_chunks:
        chunk_text, chunk_metadata = matching_chunks[0]
        section_num = chunk_metadata.get('section')
        
        logger.info(f"\nValidating metadata for section {section_num}:")
        logger.info("-" * 50)
        
        # Compare cleaned authority and source
        auth_matches = chunk_metadata.get('authority', '').strip() == expected_auth
        source_matches = chunk_metadata.get('source', '').strip() == expected_source
        
        logger.info(f"Authority matches: {auth_matches}")
        if not auth_matches:
            logger.info("Authority difference:")
            logger.info(f"Expected: {expected_auth}")
            logger.info(f"Got     : {chunk_metadata.get('authority', '')}")
        
        logger.info(f"Source matches: {source_matches}")
        if not source_matches:
            logger.info("Source difference:")
            logger.info(f"Expected: {expected_source}")
            logger.info(f"Got     : {chunk_metadata.get('source', '')}")
        
        # Print all metadata fields
        logger.info("\nAll metadata fields:")
        for key, value in chunk_metadata.items():
            if isinstance(value, str):
                logger.info(f"{key}: {value[:100]}...")
            else:
                logger.info(f"{key}: {value}")
    else:
        logger.error("No matching chunks found!")

def main():
    # Path to test XML file
    xml_path = Path("data/agencies/commerce-department/xml/title_13_chapter_III_2018-01-01.xml")
    
    try:
        validate_metadata_extraction(xml_path)
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 