# Federal Regulations Analysis Pipeline

This document describes the pipeline architecture and process flow for analyzing federal regulations.

## Pipeline Overview

The pipeline consists of five main stages, each handling a specific aspect of regulation processing:

1. Data Collection (`get_data.py`)
2. Text Processing (`process_data.py`)
3. Embedding Generation (`embed_regulations.py`)
4. Search Index Creation (`export_to_faiss.py`)
5. Analysis & Visualization (`visualize_metrics.py`)

## Pipeline Stages

### 1. Data Collection

**Module**: `pipelines/get_data.py`

Handles downloading regulations from the eCFR API:
- Downloads regulations for all agencies
- Manages content versioning
- Stores both XML and plain text formats
- Includes checkpoint system to avoid redownloading

```python
from pipelines.get_data import ECFRDownloader
downloader = ECFRDownloader()
downloader.download_all_agencies()
```

### 2. Text Processing

**Module**: `pipelines/process_data.py`

Processes downloaded regulations and calculates readability metrics:
- Calculates various readability scores
- Computes text complexity measures
- Stores results in SQLite database
- Handles agency hierarchy mapping

Key metrics include:
- Flesch Reading Ease
- Gunning Fog Index
- SMOG Index
- Type-token ratio
- Sentence complexity measures

### 3. Embedding Generation

**Module**: `pipelines/embed_regulations.py`

Creates embeddings for semantic search:
- Chunks regulations into meaningful segments
- Generates embeddings using sentence-transformers
- Preserves document structure and context
- Stores embeddings with metadata

Chunking strategy:
- Maintains section boundaries
- Preserves hierarchical structure
- Includes contextual metadata
- Handles cross-references

### 4. Search Index Creation

**Module**: `pipelines/export_to_faiss.py`

Exports embeddings to FAISS for efficient similarity search:
- Creates FAISS index from embeddings
- Stores metadata separately for quick retrieval
- Optimizes for search performance
- Maintains ID mapping for results

### 5. Analysis & Visualization

**Module**: `pipelines/visualize_metrics.py`

Generates visualizations and statistical analyses:
- Creates readability distribution plots
- Generates agency comparisons
- Produces complexity analysis charts
- Calculates statistical summaries

## Checkpoint System

The pipeline implements checkpoints at each stage to avoid reprocessing:

```python
if check_db_exists("data/db/regulations.db"):
    print("Database exists - skipping processing")
else:
    process_agencies()
```

## Directory Structure

```
data/
├── agencies/                # Downloaded regulations
│   └── {agency-slug}/
│       ├── xml/           # Raw XML content
│       └── text/          # Plain text versions
├── db/                     # SQLite databases
│   ├── regulations.db     # Processed metrics
│   └── embeddings.db      # Regulation embeddings
├── faiss/                  # Search indices
│   ├── regulation_index.faiss
│   └── regulation_metadata.json
├── plots/                  # Generated visualizations
├── stats/                  # Statistical analyses
└── logs/                   # Processing logs
```

## Error Handling

Each pipeline stage includes comprehensive error handling:
- Logs errors with context
- Maintains data consistency
- Allows for partial processing
- Enables recovery from failures

## Configuration

Pipeline settings are managed through `config/default.yml`:
- Data directories
- Model parameters
- Processing options
- Logging settings

## Running the Pipeline

The complete pipeline can be run using:

```bash
python main.py
```

Individual stages can be run separately:

```bash
python -m pipelines.get_data
python -m pipelines.process_data
python -m pipelines.embed_regulations
python -m pipelines.export_to_faiss
python -m pipelines.visualize_metrics
```

## Performance Considerations

- Implements batch processing for large datasets
- Uses efficient data structures
- Includes progress tracking
- Optimizes memory usage
- Supports parallel processing where applicable 