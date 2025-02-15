# Federal Regulations Analysis Pipeline

A Python-based tool for downloading, analyzing, and visualizing the readability of federal regulations from the Electronic Code of Federal Regulations (eCFR).

## Overview

This tool provides an automated pipeline to:
1. Download regulations from the eCFR API
2. Process and analyze text metrics (readability, complexity, etc.)
3. Generate visualizations and statistical summaries
4. Search regulations using semantic similarity
4. Search regulations using semantic similarity

## Features

- Downloads regulations from multiple federal agencies
- Calculates various readability metrics:
  - Flesch Reading Ease
  - Flesch-Kincaid Grade Level
  - Gunning Fog Index
  - SMOG Index
  - And more...
- Semantic search capabilities:
  - Natural language queries
  - Similarity-based matching
  - Relevant regulation retrieval
- Semantic search capabilities:
  - Natural language queries
  - Similarity-based matching
  - Relevant regulation retrieval
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
- `data/faiss/` - Search index and metadata

## Project Structure

```
cfr_tool/
├── main.py              # Main entry point and pipeline orchestration
├── get_data.py         # eCFR data downloading functionality
├── process_data.py     # Text processing and metrics calculation
├── visualize_metrics.py # Data visualization and reporting
├── scripts/
│   ├── search_regulations.py    # Semantic search functionality
│   ├── export_to_faiss.py      # Search index creation
│   └── regulation_embeddings/   # Embedding utilities
├── scripts/
│   ├── search_regulations.py    # Semantic search functionality
│   ├── export_to_faiss.py      # Search index creation
│   └── regulation_embeddings/   # Embedding utilities
└── data/               # Generated data and output files
    ├── logs/           # Processing logs
    ├── db/            # SQLite database
    ├── plots/         # Generated visualizations
    ├── stats/         # Statistical summaries
    └── faiss/         # Search indices and metadata
    ├── stats/         # Statistical summaries
    └── faiss/         # Search indices and metadata
```

## Dependencies

- rich - For console interface and progress tracking
- requests - For API communication
- textstat - For text analysis metrics
- pandas - For data processing
- matplotlib/seaborn - For visualization
- SQLAlchemy - For database operations
- sentence-transformers - For text embeddings
- faiss-cpu - For similarity search
- sentence-transformers - For text embeddings
- faiss-cpu - For similarity search