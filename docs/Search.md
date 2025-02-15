# Regulation Search System

The regulation search system uses semantic search with enriched embeddings to find relevant federal regulations based on natural language queries. It leverages metadata-aware embeddings and Faiss vector similarity search to match user questions with regulation chunks.

## Architecture

The search system consists of four main components:

1. **Enriched Embeddings**: Combines text embeddings with metadata embeddings for richer semantic understanding
2. **Faiss Index**: A vector similarity search index optimized for 1536-dimensional enriched vectors
3. **SQLite Database**: Stores regulation chunks, metadata, and cross-references
4. **Metadata Integration**: Enriches search with definitions, cross-references, and authority information

```mermaid
graph LR
    A[User Query] --> B[Base Embedding]
    B --> C[Enriched Vector]
    M[Metadata Fields] --> C
    C --> D[Faiss Index]
    D --> E[Similar Chunks]
    E --> F[Metadata Filtering]
    F --> G[Ranked Results]
```

## Enriched Embeddings

The system uses a composite embedding approach:

- Base text embedding (384d)
- Cross-references embedding (384d)
- Definitions embedding (384d)
- Authority/enforcement embedding (384d)

This creates a 1536-dimensional vector that captures both content and context.

### Example Enriched Search

```python
from scripts.search_regulations import RegulationSearcher

# Initialize searcher
searcher = RegulationSearcher(
    index_path="data/faiss/regulation_index.faiss",
    metadata_path="data/faiss/regulation_metadata.json",
    db_path="data/db/regulation_embeddings.db",
    model_name="all-MiniLM-L6-v2"
)

# Search with metadata filtering
results = searcher.search_similar(
    query_embedding,
    filters={
        "agency": "agriculture-department",
        "date": "2015-01-01"  # Only regulations after 2015
    }
)
```

## Search Features

### 1. Metadata-Aware Search
- Matches on related regulations through cross-references
- Understands defined terms and their relationships
- Considers enforcement context and authority

### 2. Structured Filtering
- Agency-specific search
- Date range filtering
- Section and chapter filtering
- Hierarchical navigation

### 3. Smart Ranking
- Combines text similarity with metadata relevance
- Deduplicates similar content
- Preserves regulatory context
- Minimum similarity threshold (0.2)

## Example Searches

### Cross-Reference Aware Search
```
Query: "Cotton classification requirements"

Result 1:
Score: 0.892
Agency: Department of Agriculture
Text: "Requirements for cotton classification..."
Cross-References: ["7 CFR 28.8", "7 CFR 28.9"]

Result 2: 
Score: 0.857
Agency: Department of Agriculture
Text: "Related cotton standards..."
From Cross-Reference: "7 CFR 28.8"
```

### Definition-Enhanced Search
```
Query: "Micronaire requirements"

Result 1:
Score: 0.878
Agency: Department of Agriculture
Text: "Cotton classification standards..."
Definitions: {
    "Micronaire": "A measure of cotton fiber fineness..."
}
```

## Command Line Usage

```bash
# Basic search
python scripts/search_regulations.py "FOIA request requirements"

# Agency-specific search
python scripts/search_regulations.py --agency "agriculture-department" "cotton standards"

# Date-filtered search
python scripts/search_regulations.py --after "2015-01-01" "drone regulations"
```

## Performance Optimizations

The system uses several optimizations:

1. **Vector Search**
   - FAISS index for fast similarity search
   - Batch processing of queries
   - Multi-threaded search (4 threads)

2. **Metadata Handling**
   - Cached metadata lookups
   - Pre-computed enriched embeddings
   - Efficient filtering

3. **Result Processing**
   - Early similarity filtering
   - Duplicate removal
   - Batch result formatting

## Implementation Details

The enriched embedding process:

```python
# Generate base embedding
text_embedding = model.encode(text)

# Generate metadata embeddings
cross_refs_embedding = model.encode(cross_references)
definitions_embedding = model.encode(definitions)
authority_embedding = model.encode(authority)

# Combine embeddings
enriched_embedding = np.concatenate([
    text_embedding,
    cross_refs_embedding,
    definitions_embedding,
    authority_embedding
])

# Normalize final vector
enriched_embedding = enriched_embedding / np.linalg.norm(enriched_embedding)
```

## Future Enhancements

1. **Search Capabilities**
   - Boolean query support
   - Temporal awareness
   - Agency relationship mapping

2. **Performance**
   - GPU acceleration
   - Distributed search
   - Dynamic batch sizing

3. **Metadata Integration**
   - Enhanced cross-reference tracking
   - Regulatory amendment history
   - Agency jurisdiction mapping