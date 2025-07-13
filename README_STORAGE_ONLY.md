# PDF to Milvus Storage System

A simple system for storing PDF documents in Milvus vector database. This system focuses only on **storage** - processing PDFs, generating embeddings, and storing them in Milvus. No retrieval or generation functionality is included.

## Features

- **PDF Processing**: Extract and chunk text from PDF documents
- **Vector Embeddings**: Generate embeddings using sentence-transformers
- **Milvus Storage**: Store vectors in Milvus with proper indexing
- **Batch Processing**: Store multiple PDFs from a directory
- **Statistics**: Get storage statistics and system information

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Milvus Connection

Copy `env_example.txt` to `.env` and update your Milvus settings:

```bash
MILVUS_HOST=your_milvus_host
MILVUS_PORT=19530
MILVUS_USER=your_username  # if authentication is enabled
MILVUS_PASSWORD=your_password  # if authentication is enabled
```

### 3. Store PDFs

#### Command Line Usage

```bash
# Store a single PDF
python pdf_to_milvus.py --pdf path/to/your/document.pdf

# Store all PDFs from a directory
python pdf_to_milvus.py --directory path/to/pdf/directory

# Show storage statistics
python pdf_to_milvus.py --stats

# Clear all documents
python pdf_to_milvus.py --clear
```

#### Python Usage

```python
from pdf_to_milvus import PDFToMilvus

# Initialize the system
pdf_storage = PDFToMilvus()
pdf_storage.initialize()

# Store a single PDF
success = pdf_storage.store_pdf("document.pdf", "doc_001")

# Store multiple PDFs from a directory
results = pdf_storage.store_multiple_pdfs("pdf_directory")

# Get statistics
stats = pdf_storage.get_storage_stats()

# Close the system
pdf_storage.close()
```

## Architecture

```
PDF Document → PDF Processor → Text Chunks → Embedding Generator → Milvus Vector Database
```

## Configuration

### Milvus Settings (`config.py`)

```python
# Milvus Configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "pdf_documents"
DIMENSION = 768  # For sentence-transformers/all-MiniLM-L6-v2
```

### Text Processing Settings

```python
# Text chunking configuration
CHUNK_SIZE = 1000      # Number of words per chunk
CHUNK_OVERLAP = 200    # Overlap between chunks
MAX_TEXT_LENGTH = 8192 # Maximum text length for storage
```

## API Reference

### PDFToMilvus Class

#### Methods

- `initialize()`: Connect to Milvus and create collection
- `store_pdf(pdf_path, document_id=None)`: Store a single PDF
- `store_multiple_pdfs(pdf_directory)`: Store all PDFs from a directory
- `get_storage_stats()`: Get storage statistics
- `clear_all_documents()`: Clear all documents from Milvus
- `close()`: Close the system

## Example Workflow

1. **Initialize the system**:
   ```python
   pdf_storage = PDFToMilvus()
   pdf_storage.initialize()
   ```

2. **Store PDF documents**:
   ```python
   # Single PDF
   pdf_storage.store_pdf("document1.pdf", "doc_001")
   
   # Multiple PDFs
   pdf_storage.store_multiple_pdfs("pdf_directory")
   ```

3. **Check storage statistics**:
   ```python
   stats = pdf_storage.get_storage_stats()
   print(f"Total chunks stored: {stats['milvus_stats']['row_count']}")
   ```

## Command Line Examples

### Store a Single PDF
```bash
python pdf_to_milvus.py --pdf my_document.pdf
```

### Store All PDFs in a Directory
```bash
python pdf_to_milvus.py --directory ./my_pdfs/
```

### Check Storage Statistics
```bash
python pdf_to_milvus.py --stats
```

### Clear All Documents
```bash
python pdf_to_milvus.py --clear
```

## File Structure

```
├── pdf_to_milvus.py      # Main storage system
├── simple_example.py      # Simple usage example
├── milvus_client.py       # Milvus operations
├── pdf_processor.py       # PDF text extraction and chunking
├── embedding_generator.py # Vector embedding generation
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
└── README_STORAGE_ONLY.md # This file
```

## Troubleshooting

### Common Issues

1. **Connection to Milvus failed**:
   - Check if Milvus is running
   - Verify host and port settings in `.env`
   - Check authentication credentials if enabled

2. **PDF processing failed**:
   - Ensure the PDF file exists and is readable
   - Check if the PDF contains extractable text
   - Verify PyMuPDF installation

3. **Embedding generation failed**:
   - Check internet connection (for model download)
   - Verify sentence-transformers installation
   - Check available memory

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

- **Chunk Size**: 500-1000 words per chunk works well for most documents
- **Overlap**: 10-20% overlap helps maintain context
- **Batch Processing**: Use `store_multiple_pdfs()` for multiple documents
- **Memory**: Large PDFs may require more memory for processing

## What's NOT Included

This system focuses only on storage. The following features are **not** included:

- ❌ Document retrieval/search
- ❌ Question answering
- ❌ Text generation
- ❌ Query processing
- ❌ Result ranking

If you need these features, you can extend the system or use the full RAG framework. 