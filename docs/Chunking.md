# Regulation Chunking Strategy

## Overview
The regulation chunking system splits XML regulation documents into meaningful segments while preserving semantic context and metadata. The system uses a hierarchical approach based on the XML structure of federal regulations.

## Chunking Components

### 1. XML Structure
The chunker recognizes standard regulation document divisions:
```python
DIVISION_TAGS = {
    "DIV1": "Title",
    "DIV2": "Chapter",
    "DIV3": "Part",
    "DIV4": "Subpart",
    "DIV5": "Section",
    "DIV6": "Subsection",
    "DIV7": "Paragraph",
    "DIV8": "Subparagraph"
}
```

### 2. Metadata Extraction
Each chunk preserves important metadata:
- Section numbers and titles
- Cross-references
- Term definitions
- Authority/enforcement information
- Hierarchy information
- Last revision dates
- Regulatory intent/purpose

### 3. Chunking Parameters
- Maximum chunk length: 1000 characters
- Minimum chunk length: 100 characters
- Default XML depth: DIV8 (Subparagraph level)

## Chunking Process

1. **Document Parsing**
```python
def parse_xml(document_path: Path) -> etree._Element:
    """
    Parse XML with robust error handling and recovery mode.
    - Uses custom parser with recovery mode
    - Logs parsing issues
    - Handles malformed XML gracefully
    """
```

2. **Hierarchy Extraction**
```python
def extract_hierarchy_metadata(element: etree._Element) -> Dict:
    """
    Build document hierarchy:
    - First tries to parse existing hierarchy metadata
    - Falls back to building from element structure
    Returns: {
        "title": "42",
        "chapter": "IV",
        "part": "403",
        "section": "1"
    }
    """
```

3. **Section Metadata Extraction**
```python
def extract_section_metadata(element: etree._Element) -> Dict:
    """
    Extract metadata fields:
    - Section number and title
    - Authority and source from parent DIV5
    - Cross-references
    - Term definitions
    - Enforcement agencies
    - Last revision date
    - Regulatory intent
    """
```

4. **Text Chunking**
```python
def iter_chunks(element: etree._Element, metadata: Dict) -> Iterator[Tuple[str, Dict]]:
    """
    Split text while preserving:
    - Paragraph boundaries
    - Paragraph identifiers
    - Minimum/maximum chunk lengths
    - Metadata association
    """
```

## Chunking Rules

1. **Boundary Preservation**
- Respects paragraph boundaries
- Preserves paragraph identifiers
- Maintains logical divisions
- Avoids splitting mid-sentence

2. **Context Preservation**
- Includes section headers
- Maintains paragraph numbering
- Preserves hierarchical context
- Keeps related content together

3. **Text Cleaning**
- Removes extra whitespace
- Normalizes quotes and dashes
- Handles special characters
- Maintains consistent formatting

## Example Chunk

```json
{
    "text": "§ 404.1520 Evaluation of disability in general. (a) Steps in evaluating disability. We consider all evidence in your case record when we make a determination or decision whether you are disabled...",
    "metadata": {
        "section": "404.1520",
        "title": "20",
        "chapter": "III",
        "hierarchy": {
            "title": "20",
            "chapter": "III",
            "part": "404",
            "section": "1520"
        },
        "cross_references": [
            "§ 404.1512",
            "§ 404.1513"
        ],
        "definitions": [
            "disability",
            "determination"
        ],
        "authority": "42 U.S.C. 405(a)"
    }
}
```

## Performance Considerations

1. **Memory Management**
- Processes chunks in small batches
- Clears SQLAlchemy session after batches
- Uses streaming for large files
- Implements efficient text cleaning

2. **Error Handling**
- Recovers from malformed XML
- Logs parsing issues
- Continues processing on errors
- Maintains partial results

3. **Optimization**
- Uses custom XML parser
- Implements efficient text cleaning
- Processes in configurable batch sizes
- Minimizes memory usage

## Usage Example

```python
from regulation_embeddings.chunkers import XMLChunker

# Initialize chunker
chunker = XMLChunker(
    max_chunk_length=1000,
    xml_tag_depth=".//DIV8",
    min_chunk_length=100
)

# Process document
chunks = chunker.chunk_document(document_path)

# Access chunks with metadata
for chunk_text, metadata in chunks:
    print(f"Section: {metadata.get('section')}")
    print(f"Text length: {len(chunk_text)}")
    print(f"Cross-references: {metadata.get('cross_references', [])}")
``` 