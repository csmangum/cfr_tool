# Enriched Embeddings with Metadata

This document explains how the regulation search system uses enriched embeddings that combine text content with metadata to improve search relevance.

## Overview

The system creates enriched embeddings by concatenating multiple embedding vectors:

1. Base text embedding (384 dimensions)
2. Cross-reference embedding (384 dimensions) 
3. Definition embedding (384 dimensions)
4. Authority embedding (384 dimensions)

Total dimension: 1536 (4 x 384)

## Components

### 1. Base Text Embedding
- Generated from the main regulation text content
- Captures the core semantic meaning of the regulation
- Uses the base all-MiniLM-L6-v2 model (384 dimensions)

### 2. Cross-Reference Embedding 
- Generated from related regulation references
- Helps connect related regulations during search
- Example: References to other sections, chapters, or titles
- Enables finding regulations that reference similar content

### 3. Definition Embedding
- Generated from defined terms and their explanations
- Captures domain-specific terminology
- Helps match queries using either formal or informal language
- Improves search when users don't know exact regulatory terms

### 4. Authority Embedding
- Generated from enforcement agency information
- Captures which agencies have authority over regulations
- Helps match queries about specific agencies' rules
- Improves agency-specific searches

## Implementation

The enriched embeddings are created in `scripts/regulation_embeddings/embedders.py`:

```python
def embed_text_with_metadata(self, text: str, metadata: Dict[str, List[str]]) -> np.ndarray:
    # Get base text embedding
    text_embedding = self.model.encode(text)
    
    # Generate embeddings for metadata fields
    cross_refs = " ".join(metadata.get("cross_references", []))
    definitions = " ".join(metadata.get("definitions", []))
    authority = " ".join(metadata.get("enforcement_agencies", []))
    
    # Create embeddings for metadata (or zeros if empty)
    cross_refs_embedding = self.model.encode(cross_refs) if cross_refs else np.zeros(384)
    definitions_embedding = self.model.encode(definitions) if definitions else np.zeros(384)
    authority_embedding = self.model.encode(authority) if authority else np.zeros(384)
    
    # Concatenate all embeddings
    return np.concatenate([
        text_embedding,
        cross_refs_embedding, 
        definitions_embedding,
        authority_embedding
    ])
```

## Benefits

1. **Improved Search Relevance**
   - Matches on both content and metadata
   - Finds related regulations through cross-references
   - Understands domain terminology through definitions

2. **Flexible Querying**
   - Users can search using formal or informal language
   - Queries can target specific agencies or authorities
   - System understands relationships between regulations

3. **Richer Context**
   - Each embedding vector captures different aspects of the regulation
   - Metadata enrichment provides additional search signals
   - Helps disambiguate similar regulations

## Storage and Retrieval

The enriched embeddings are:
- Stored in SQLite as binary data
- Indexed in FAISS for fast similarity search
- Normalized to unit length for cosine similarity
- Total size: 1536 dimensions x 4 bytes = 6.144 KB per embedding

## Example

For a regulation about food safety:

1. **Base Text**: "Food manufacturers must maintain sanitary conditions..."
2. **Cross-References**: "See 21 CFR 110.10 for personnel requirements"
3. **Definitions**: "Food manufacturer: An establishment that processes..."
4. **Authority**: "FDA, Department of Health and Human Services"

The system combines embeddings from all these components to create a rich representation that can match queries about food safety, FDA regulations, manufacturing requirements, or related topics.

## Search Process

1. Query embedding is created with zero vectors for metadata components
2. Similarity search compares against enriched embeddings
3. Metadata components influence matching without requiring exact matches
4. Results are ranked by overall similarity across all components

## Future Improvements

Potential enhancements to the enriched embedding system:

1. Add weights to different components
2. Include more metadata fields
3. Use specialized models for different components
4. Implement dynamic component selection based on query type