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

## Semantic Segmentation

### Overview
The chunking process now includes semantic segmentation using NLP techniques to enhance the quality of chunked regulatory text. This approach ensures that chunks are semantically coherent and contextually meaningful.

### Components

1. **Sentence Embeddings**
   - Use Sentence Transformers (e.g., `all-MiniLM-L6-v2`) to generate embeddings for each sentence.
   - Compute cosine similarity between adjacent sentences to quantify their semantic relationship.

2. **Chunk Boundaries**
   - Define a threshold (e.g., 0.7): If similarity between two sentences drops below this threshold, insert a segmentation boundary.
   - Experiment with different thresholds and fine-tune for optimal chunk sizes.

3. **Advanced Techniques**
   - Optionally, incorporate TextTiling or BERT-based segmentation models for improved topic shift detection.

### Process

1. **Compute Sentence Embeddings**
```python
def compute_sentence_embeddings(sentences: List[str]) -> np.ndarray:
    """Compute sentence embeddings using Sentence Transformers."""
    return model.encode(sentences, convert_to_numpy=True)
```

2. **Identify Chunk Boundaries**
```python
def identify_chunk_boundaries(sentences: List[str]) -> List[int]:
    """Identify chunk boundaries based on semantic similarity."""
    embeddings = compute_sentence_embeddings(sentences)
    similarities = np.array(
        [np.dot(embeddings[i], embeddings[i + 1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]))
         for i in range(len(embeddings) - 1)]
    )
    boundaries = [i + 1 for i, sim in enumerate(similarities) if sim < similarity_threshold]
    return boundaries
```

3. **Chunk Formation**
```python
def form_chunks(sentences: List[str], boundaries: List[int]) -> List[str]:
    """Form chunks based on identified boundaries."""
    chunks = []
    start = 0
    for boundary in boundaries:
        chunk_text = " ".join(sentences[start:boundary])
        if len(chunk_text) >= min_chunk_length:
            chunks.append(chunk_text)
        start = boundary
    # Add remaining sentences as the last chunk
    chunk_text = " ".join(sentences[start:])
    if len(chunk_text) >= min_chunk_length:
        chunks.append(chunk_text)
    return chunks
```

### Example

```python
# Example sentences
sentences = [
    "The regulation requires all manufacturers to maintain records.",
    "These records must be kept for a minimum of five years.",
    "Failure to comply with this requirement may result in penalties.",
    "Penalties include fines and suspension of licenses.",
    "Manufacturers must also submit annual reports to the agency."
]

# Compute embeddings and identify boundaries
embeddings = compute_sentence_embeddings(sentences)
boundaries = identify_chunk_boundaries(sentences)

# Form chunks
chunks = form_chunks(sentences, boundaries)

# Output chunks
for chunk in chunks:
    print(chunk)
```

### Benefits

1. **Detection of Natural Topic Boundaries**
   - By using sentence embeddings and cosine similarity, the system can detect significant topic shifts within the text. This ensures that chunks are created at points where the content naturally changes, leading to more coherent and contextually meaningful chunks.

2. **Reduction of Concept Fragmentation**
   - Semantic segmentation helps in grouping related sentences together, reducing the chances of splitting key concepts across multiple chunks. This makes the chunks more useful for downstream applications like similarity search and analysis.

3. **Improved Retrieval Accuracy**
   - With contextually relevant chunks, the embedding-based similarity search will be more accurate. This is because the chunks will better represent the underlying topics and concepts, leading to more precise search results.

### Evaluation

1. **Semantic Similarity**
   - Use sentence embeddings and cosine similarity to ensure that sentences within a chunk are semantically related. This will help in detecting natural topic boundaries and maintaining coherence within each chunk.

2. **Contextual Relevance**
   - Each chunk should be contextually meaningful and retain important metadata such as section headers. This will ensure that the chunks are useful for downstream applications like similarity search and analysis.

3. **Reduction of Concept Fragmentation**
   - Aim to group related sentences together to reduce the chances of splitting key concepts across multiple chunks. This will make the chunks more coherent and easier to analyze.

4. **Human Review**
   - Conduct human reviews to evaluate the coherence of the chunks. This will involve assessing whether the chunks make sense as standalone units and whether they accurately represent the underlying topics and concepts.

5. **Automatic Metrics**
   - Use automatic metrics such as Rouge and BLEU scores to evaluate the internal consistency and coherence of the chunks. These metrics will provide quantitative measures of chunk quality.
