"""
Ultra Memory Optimized Configuration
Settings designed for processing very large PDFs (500+ pages) with minimal memory usage
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Ultra-conservative memory settings
ULTRA_MEMORY_MB = 5000  # Maximum memory usage target
ULTRA_PAGES_PER_BATCH = 3  # Process only 3 pages at a time
ULTRA_BATCH_SIZE = 10  # Small batch size for embeddings
ULTRA_CHUNK_SIZE = 1000  # Smaller chunks to reduce memory per chunk
ULTRA_CHUNK_OVERLAP = 200  # Reduced overlap

# Milvus Configuration
MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
MILVUS_USER = os.getenv('MILVUS_USER', '')
MILVUS_PASSWORD = os.getenv('MILVUS_PASSWORD', '')

# Collection Configuration
COLLECTION_NAME = "pdf_documents"

INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "COSINE"
NPROBE = 10

# Text Processing Configuration (Ultra-conservative)
CHUNK_SIZE = ULTRA_CHUNK_SIZE
CHUNK_OVERLAP = ULTRA_CHUNK_OVERLAP
MAX_TEXT_LENGTH = 4096  # Reduced from 8192

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OpenAI Configuration
USE_OPENAI_EMBEDDINGS = os.getenv('USE_OPENAI_EMBEDDINGS', 'false').lower() == 'true'
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'text-embedding-3-small')

# Dimension Configuration
# For sentence-transformers/all-MiniLM-L6-v2: 384
# For OpenAI text-embedding-3-small: 1536
# For OpenAI text-embedding-ada-002: 1536 

DIMENSION = 1536 if USE_OPENAI_EMBEDDINGS else 384

# Processing delays (in seconds)
PAGE_PROCESSING_DELAY = 0.01  # Delay between pages
BATCH_PROCESSING_DELAY = 0.05  # Delay between batches
FILE_PROCESSING_DELAY = 2.0  # Delay between files

# Memory monitoring settings
MEMORY_CHECK_INTERVAL = 10  # Check memory every N pages
MEMORY_WARNING_THRESHOLD = 0.8  # Warn when memory usage exceeds 80% of target

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = "logs/ultra_optimized_processing.log"

# Error handling
MAX_RETRIES = 3  # Maximum retries for failed operations
RETRY_DELAY = 1.0  # Delay between retries (seconds) 