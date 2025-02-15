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
- Maximum chunk length: 1000 tokens
- Minimum chunk length: 100 tokens
- Default XML depth: DIV8 (Subparagraph level)

## Chunking Process

1. **Document Parsing**
```python
def parse_xml(document_path: Path) -> etree._Element:
    """
    Parse XML with robust error handling and recovery mode.
    Preserves document structure and handles malformed XML.
    """
```

2. **Hierarchy Extraction**
```python
def extract_hierarchy_metadata(element: etree._Element) -> Dict:
    """
    Build document hierarchy from:
    - Explicit hierarchy metadata if available
    - Element structure as fallback
    Returns: {
        "title": "42",
        "chapter": "IV",
        "part": "403",
        "section": "1"
    }
    """
```

3. **Semantic Metadata Extraction**
```python
def extract_section_metadata(element: etree._Element) -> Dict:
    """
    Extract semantic fields:
    - Cross-references
    - Term definitions
    - Authority information
    - Revision dates
    - Regulatory intent
    """
```

4. **Text Chunking**
```python
def iter_chunks(element: etree._Element, metadata: Dict) -> Iterator[Tuple[str, Dict]]:
    """
    Split text while preserving:
    - Paragraph boundaries
    - Semantic coherence
    - Section context
    - Metadata association
    """
```

## Chunking Rules

1. **Boundary Preservation**
- Respect paragraph and section boundaries
- Never split mid-sentence
- Maintain list item grouping
- Keep related subsections together

2. **Context Preservation**
- Include section headers with chunks
- Preserve paragraph numbering
- Maintain hierarchical context
- Keep definition terms with their definitions

3. **Metadata Enrichment**
- Attach relevant cross-references
- Include applicable definitions
- Link to authority information
- Preserve revision history

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
- Stream large XML files
- Process chunks in batches
- Clear parsed elements after use
- Use iterative parsing for large documents

2. **Error Handling**
- Recover from malformed XML
- Skip invalid sections
- Log parsing errors
- Maintain partial results on failure

3. **Optimization**
- Cache frequently used XPaths
- Reuse parsed elements
- Batch metadata extraction
- Minimize string operations

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
    print(f"Section: {metadata['section']}")
    print(f"Text length: {len(chunk_text)}")
    print(f"Cross-references: {metadata['cross_references']}")
``` 