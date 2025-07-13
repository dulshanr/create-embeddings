#!/usr/bin/env python3
"""
Clear and recreate collection with correct dimension.
"""

import os
from dotenv import load_dotenv
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from config import *

# Load environment variables
load_dotenv()

def clear_and_recreate():
    """Clear existing collection and recreate with correct dimension"""
    
    print("=== Clearing and Recreating Collection ===\n")
    
    try:
        # Get connection settings from environment
        host = os.getenv('MILVUS_HOST', 'localhost')
        port = os.getenv('MILVUS_PORT', '19530')
        user = os.getenv('MILVUS_USER', '')
        password = os.getenv('MILVUS_PASSWORD', '')
        
        # Connect to Milvus
        if user and password:
            connections.connect("default", host=host, port=port, user=user, password=password)
            print(f"✓ Connected to Milvus at {host}:{port} with authentication")
        else:
            connections.connect("default", host=host, port=port)
            print(f"✓ Connected to Milvus at {host}:{port}")
        
        collection_name = COLLECTION_NAME
        
        # Drop existing collection if it exists
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"✓ Dropped existing collection: {collection_name}")
        else:
            print(f"Collection '{collection_name}' does not exist")
        
        # Create new collection with correct dimension
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
        ]
        
        schema = CollectionSchema(fields=fields, description="PDF document chunks with embeddings")
        collection = Collection(name=collection_name, schema=schema)
        print(f"✓ Created collection: {collection_name} with dimension {DIMENSION}")
        
        # Create index
        index_params = {
            "metric_type": METRIC_TYPE,
            "index_type": INDEX_TYPE,
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("✓ Created index on embedding field")
        
        # Test insert with correct dimension
        test_document = {
            "document_id": "test_doc_001",
            "page_number": 1,
            "chunk_id": 1,
            "text": "This is a test document chunk.",
            "embedding": [0.1] * DIMENSION  # Use correct dimension
        }
        
        collection.insert([test_document])
        collection.flush()
        print("✓ Successfully inserted test document")
        
        # Get stats
        stats = collection.get_statistics()
        print(f"✓ Collection stats: {stats}")
        
        connections.disconnect("default")
        print("✓ Disconnected from Milvus")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = clear_and_recreate()
    if success:
        print("\n✓ Collection cleared and recreated successfully!")
        print("You can now run your PDF processing again.")
    else:
        print("\n✗ Failed to clear and recreate collection") 