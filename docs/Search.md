# Regulation Search System

The regulation search system uses semantic search with sentence transformer embeddings to find relevant federal regulations based on natural language queries.

## Architecture

The search system consists of three main components:

1. **Text Embeddings**: Generates semantic vectors from regulation text (384 dimensions) using SentenceTransformer
2. **Faiss Index**: A vector similarity search index optimized for fast similarity search
3. **SQLite Database**: Stores regulation chunks and metadata

```mermaid
graph LR
    A[User Query] --> B[Query Embedding]
    B --> C[Faiss Index]
    C --> D[Similar Chunks]
    D --> E[Metadata Filtering]
    E --> F[Ranked Results]
```

## Current Implementation

The system uses the all-MiniLM-L6-v2 model to generate 384-dimensional embeddings:

```python
from scripts.search_regulations import RegulationSearcher

# Initialize searcher
searcher = RegulationSearcher(
    index_path="data/faiss/regulation_index.faiss",
    metadata_path="data/faiss/regulation_metadata.json",
    db_path="data/db/regulation_embeddings.db",
    model_name="all-MiniLM-L6-v2"
)

# Search regulations
results = searcher.search(
    query="What are the requirements for filing a FOIA request?",
    n_results=5
)
```

## Search Features

### 1. Semantic Search
- Uses sentence transformer embeddings for semantic understanding
- Matches based on meaning rather than just keywords
- Handles natural language queries effectively

### 2. Metadata Storage
- Agency information
- Title and chapter numbers
- Section numbers
- Publication dates
- Document hierarchy

### 3. Result Ranking
- Similarity scores based on cosine similarity
- Deduplication of similar content
- Minimum similarity threshold (0.2)
- Returns top N most relevant results

## Example Usage

### Command Line Interface
```bash
# Basic search with default parameters
python scripts/search_regulations.py "FOIA request requirements"

# Specify number of results
python scripts/search_regulations.py --num-results 10 "drone regulations"

# Save results to file
python scripts/search_regulations.py --save "workplace safety requirements"
```

### Interactive Mode
```bash
python scripts/search_regulations.py
# Enter queries interactively
# Press Enter for random sample questions
# Type 'quit' to exit
```

### Streamlit Interface
```bash
streamlit run scripts/streamlit_search.py
```

## Performance Optimizations

The system implements several optimizations:

1. **Vector Search**
   - FAISS IndexIDMap with FlatL2 base
   - Efficient similarity computations
   - Multi-threaded search (4 threads)

2. **Result Processing**
   - LRU cache for chunk text (1000 entries)
   - Early similarity filtering
   - Efficient deduplication

3. **Database Access**
   - SQLite for metadata storage
   - Batch processing where applicable
   - Connection pooling

## Implementation Details

The search process:

```python
def search(self, query: str, n_results: int = 5) -> list:
    """Search for relevant regulation chunks."""
    # Create query embedding
    query_embedding = self._enrich_query_embedding(query)

    # Search with expanded results for filtering
    distances, indices = self.index.search(
        query_embedding.reshape(1, -1), 
        n_results * 2
    )

    results = []
    seen_texts = set()

    # Process and filter results
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1 or distance < 0.2:  # Early filtering
            continue

        result_metadata = self.metadata.get(str(idx))
        if not result_metadata:
            continue

        chunk_text = self._load_chunk_text(idx)
        if chunk_text in seen_texts:
            continue

        seen_texts.add(chunk_text)
        results.append((result_metadata, 1/(1 + distance), chunk_text))

    # Sort by similarity score
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n_results]
```

## Future Enhancements

1. **Metadata-Enriched Embeddings**
   - Add cross-reference awareness
   - Include definition context
   - Incorporate authority information

2. **Search Capabilities**
   - Agency-specific filtering
   - Date range filtering
   - Boolean query support

3. **Performance**
   - GPU acceleration
   - Distributed search
   - Improved caching strategies
