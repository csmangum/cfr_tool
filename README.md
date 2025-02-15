# Federal Regulations Analysis Pipeline

A Python-based tool for downloading, analyzing, and visualizing the readability of federal regulations from the Electronic Code of Federal Regulations (eCFR).

## Overview

This tool provides an automated pipeline to:
1. Download regulations from the eCFR API
2. Process and analyze text metrics (readability, complexity, etc.)
3. Generate visualizations and statistical summaries
4. Search regulations using semantic similarity with enriched embeddings

## Features

- Downloads regulations from multiple federal agencies
- Intelligent document chunking:
  - Hierarchical XML parsing
  - Semantic metadata extraction
  - Context preservation
  - See [Chunking Documentation](docs/Chunking.md)
- Enriched embeddings with metadata:
  - Base text embeddings
  - Cross-reference embeddings [PLANNED]
  - Definition embeddings [PLANNED]
  - Authority embeddings [PLANNED]
  - See [Enriched Embeddings Documentation](docs/EnrichedEmbeddings.md)
- Readability metrics:
  - Flesch Reading Ease
  - Flesch-Kincaid Grade Level
  - Gunning Fog Index
  - SMOG Index
  - And more...
  - See [Readability Metrics Documentation](docs/ReadabilityMetrics.md)
- Semantic search capabilities:
  - Natural language queries
  - Similarity-based matching
  - Metadata-enriched results
  - See [Search Documentation](docs/Search.md)
- Generates visualizations and statistical reports
- Progress tracking with rich console interface
- Automated data organization and storage

## Installation

```bash
# Clone the repository
git clone https://github.com/csmangum/cfr_tool.git
cd cfr_tool

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the main script to start the analysis pipeline:

```bash
python main.py
```

The tool will:
1. Download regulations from eCFR
2. Process and analyze text metrics
3. Generate visualizations and statistics

### Searching Regulations

Search through regulations using natural language queries:

```bash
# Basic search
python scripts/search_regulations.py "What are the requirements for filing a FOIA request?"

# Interactive mode
python scripts/search_regulations.py

# Save results to file
python scripts/search_regulations.py --save "How are endangered species protected?"
```

See [Search Documentation](docs/Search.md) for detailed information about the search functionality.

## Output

Generated files will be organized in the following directories:
- `data/plots/` - Visualization plots
- `data/stats/` - Statistical summaries
- `data/logs/` - Processing logs
- `data/db/` - Database files
- `data/faiss/` - Search index and metadata

## Project Structure

```
cfr_tool/
├── main.py              # Main entry point and pipeline orchestration
├── docs/               # Documentation
│   ├── Model.md       # Model architecture details
│   ├── Search.md      # Search functionality guide
│   └── Chunking.md    # Document chunking strategy
├── scripts/
│   ├── get_data.py    # eCFR data downloading
│   ├── process_data.py # Text processing and metrics
│   ├── search_regulations.py # Semantic search
│   ├── export_to_faiss.py   # Search index creation
│   └── regulation_embeddings/
│       ├── chunkers.py      # Document chunking
│       ├── embedders.py     # Text embedding
│       ├── pipeline.py      # Processing pipeline
│       └── vector_stores.py # Vector similarity search
└── data/               # Generated data and output
    ├── logs/          # Processing logs
    ├── db/            # SQLite database
    ├── plots/         # Visualizations
    ├── stats/         # Statistical summaries
    └── faiss/         # Search indices
```

## Dependencies

- rich - Console interface and progress tracking
- requests - API communication
- textstat - Text analysis metrics
- pandas - Data processing
- matplotlib/seaborn - Visualization
- SQLAlchemy - Database operations
- sentence-transformers - Text embeddings
- faiss-cpu - Similarity search
- lxml - XML processing