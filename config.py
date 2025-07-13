import os
from dotenv import load_dotenv

load_dotenv()

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

# Text Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TEXT_LENGTH = 8192

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