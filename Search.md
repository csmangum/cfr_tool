# Regulation Search System

The regulation search system uses semantic search to find relevant federal regulations based on natural language queries. It leverages sentence embeddings and Faiss vector similarity search to match user questions with regulation chunks.

## Architecture

The search system consists of three main components:

1. **Sentence Embeddings**: Uses the `sentence-transformers` model to convert text into dense vector representations
2. **Faiss Index**: A vector similarity search index that enables efficient retrieval of relevant regulation chunks
3. **SQLite Database**: Stores the regulation chunks and their metadata

```mermaid
graph LR
    A[User Query] --> B[Sentence Transformer]
    B --> C[Query Embedding]
    C --> D[Faiss Index]
    D --> E[Similar Chunks]
    E --> F[Metadata Lookup]
    F --> G[Formatted Results]
```

## Usage

### Command Line Interface

Search for regulations using the command line:

```bash
# Basic search
python scripts/search_regulations.py "What are the requirements for filing a FOIA request?"

# Specify number of results
python scripts/search_regulations.py --num-results 3 "How are endangered species protected?"

# Save results to file
python scripts/search_regulations.py --save "What are the workplace safety requirements?"
```

### Interactive Mode

If you run the script without a query, it enters interactive mode:

```bash
python scripts/search_regulations.py
```

This allows you to:
- Enter multiple queries in succession
- Press Enter for a random sample question
- Type 'quit' to exit

### Python API

You can also use the search functionality in your Python code:

```python
from scripts.search_regulations import RegulationSearcher

# Initialize the searcher
searcher = RegulationSearcher(
    index_path="data/faiss/regulation_index.faiss",
    metadata_path="data/faiss/regulation_metadata.json",
    db_path="data/db/regulation_embeddings.db",
    model_name="all-MiniLM-L6-v2"
)

# Perform a search
results = searcher.search(
    query="What are the requirements for drone operations?",
    n_results=5
)

# Process results
for metadata, score, chunk_text in results:
    print(f"Score: {score:.3f}")
    print(f"Agency: {metadata['agency']}")
    print(f"Text: {chunk_text}\n")
```

## Example Searches and Results

Here are some example searches and their results:

### FOIA Request Requirements

```
Query: "What are the requirements for filing a FOIA request?"

Result 1:
Score: 0.892
Agency: Department of Justice
Title 28, Chapter 16, Section 16.3
Text: Requirements for making a FOIA request. (1) A request for DOJ records 
under the FOIA must be made in writing and received by mail, delivery service, 
or electronic submission. (2) The request should clearly state that it is being 
made under the FOIA...
```

### Endangered Species Protection

```
Query: "How are endangered species protected?"

Result 1:
Score: 0.857
Agency: Fish and Wildlife Service
Title 50, Chapter I, Section 17.21
Text: Endangered species are protected from take, which includes harming, 
harassing, collecting, or killing. No person shall take endangered wildlife 
within the United States, within the territorial sea of the United States...
```

## Configuration Options

The search system can be configured with several parameters:

- `--model`: Choose different sentence transformer models (default: "all-MiniLM-L6-v2")
- `--num-results`: Number of results to return (default: 5)
- `--index`: Path to the Faiss index file
- `--metadata`: Path to the metadata JSON file
- `--db`: Path to the SQLite database
- `--save`: Save results to a file

## Performance Considerations

The system uses several optimizations:
- Faiss for efficient similarity search
- Normalized embeddings for better matching
- Duplicate removal
- Minimum similarity threshold (0.2)
- Metadata caching

Results are ranked by cosine similarity score, with higher scores indicating better matches to the query.

# Semantic Search with Enriched Embeddings

The search system uses enriched embeddings that combine the base text representation with metadata-specific embeddings:

## Embedding Components
- Base text embedding (384d)
- Cross-references embedding (384d)
- Definitions embedding (384d)
- Authority/enforcement embedding (384d)

## Search Process
1. Query text is embedded using the base model
2. Query embedding is enriched with zero vectors for metadata fields
3. FAISS performs similarity search using the full 1536d vectors
4. Results are filtered and ranked based on similarity scores

## Metadata Enrichment
The system enriches embeddings with semantic metadata to improve search relevance:
- Cross-references help connect related regulations
- Definitions improve term matching
- Authority information helps match enforcement context