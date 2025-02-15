# Regulation Search Model Architecture

## Overview
The regulation search system currently uses dense retrieval with base text embeddings. Future enhancements will add metadata enrichment to provide more contextual search results.

## Core Components

### 1. Embedding Model
- **Base Model**: SentenceTransformer ('all-MiniLM-L6-v2')
- **Vector Dimension**: 384 dimensions (text only)
- **Future Enhancement**: Will expand to 1536 dimensions with metadata enrichment
- **Normalization**: L2 normalization for cosine similarity

### 2. Metadata Enrichment
The model enriches text embeddings with metadata-specific embeddings:

```python
enriched_embedding = np.concatenate([
    text_embedding,           # Base text embedding (384d)
    cross_refs_embedding,     # Cross-references (384d)
    definitions_embedding,    # Term definitions (384d)
    authority_embedding       # Enforcement agencies (384d)
])
```

### 3. Vector Storage
- **Index Type**: FAISS IndexIDMap with FlatL2 base
- **Distance Metric**: L2 distance (converted from cosine similarity)
- **Dimension**: 1536 (4 x 384 for enriched embeddings)
- **Batch Processing**: 32 vectors per batch
- **Multithreading**: Enabled with 4 threads

## Search Process

1. **Query Processing**
```python
# Convert query to embedding
query_embedding = model.encode(query).astype(np.float32)

# Normalize for cosine similarity
faiss.normalize_L2(query_embedding)

# Search with expanded results
distances, indices = index.search(query_embedding, n_results * 2)
```

2. **Result Filtering**
- Removes duplicate content
- Filters by similarity threshold (0.2)
- Deduplicates by content hash
- Returns top N unique results

## Metadata Integration

### Extracted Metadata
- Citations and Cross-References
- Enforcement Agencies
- Term Definitions
- Last Revision Dates
- Regulatory Intent/Purpose

### Storage Schema
```sql
CREATE TABLE regulation_chunks (
    id INTEGER PRIMARY KEY,
    agency TEXT,
    title TEXT,
    chapter TEXT,
    date TEXT,
    chunk_text TEXT,
    chunk_index INTEGER,
    embedding BLOB,
    section TEXT,
    hierarchy TEXT
);

-- Indices for efficient filtering
CREATE INDEX idx_agency ON regulation_chunks (agency);
CREATE INDEX idx_date ON regulation_chunks (date);
CREATE INDEX idx_section ON regulation_chunks (section);
```

## Performance Optimizations

### 1. Batch Processing
- Query batching (32 queries per batch)
- Embedding generation batching
- Database operations batching (50 records per batch)

### 2. Memory Management
- LRU cache for chunk text (1000 entries)
- Session cleanup after batch operations
- Streaming results for large queries

### 3. Search Optimization
```python
# Expanded search with post-filtering
distances, indices = self.index.search(
    query_embedding, 
    n_results * 2  # Request more results for filtering
)

# Efficient deduplication
seen_texts = set()
results = []

for distance, idx in zip(distances, indices):
    if idx == -1 or distance < 0.2:  # Early filtering
        continue
        
    chunk_text = self._load_chunk_text(idx)
    if chunk_text in seen_texts:
        continue
        
    seen_texts.add(chunk_text)
    results.append((metadata, distance, chunk_text))
```

## Usage Example

```python
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

# Format and display results
for metadata, score, chunk_text in results:
    print(f"Score: {score:.3f}")
    print(f"Agency: {metadata['agency']}")
    print(f"Title {metadata['title']}, Chapter {metadata['chapter']}")
    print(f"Text:\n{chunk_text}\n")
```

## Model Limitations

1. **Context Window**
   - Limited to 1000 tokens per chunk
   - May split related content across chunks

2. **Metadata Coverage**
   - Not all regulations contain complete metadata
   - Some fields may be missing or incomplete

3. **Search Precision**
   - Semantic similarity may not capture exact legal meanings
   - Requires post-processing for highly specific queries

4. **Resource Requirements**
   - FAISS index loads full vector space into memory
   - Large batch sizes may require significant RAM

## Future Improvements

1. **Model Enhancements**
   - Fine-tuning on legal domain text
   - Larger context windows
   - Hierarchical embeddings

2. **Search Features**
   - Boolean filters
   - Date range filtering
   - Agency-specific search

3. **Performance**
   - GPU acceleration
   - Distributed search
   - Improved caching