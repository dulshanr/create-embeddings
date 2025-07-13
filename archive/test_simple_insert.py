#!/usr/bin/env python3
"""
Simple test to debug Milvus insert issue.
"""

import os
from dotenv import load_dotenv
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from config import *

# Load environment variables
load_dotenv()

def test_simple_insert():
    """Test a very simple insert with proper debugging"""
    
    print("=== Simple Milvus Insert Test ===\n")
    
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
        
        # Drop existing collection if it exists
        collection_name = COLLECTION_NAME
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"✓ Dropped existing collection: {collection_name}")
        
        # Create new collection with correct schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
        ]
        
        schema = CollectionSchema(fields=fields, description="Test collection")
        collection = Collection(name=collection_name, schema=schema)
        print(f"✓ Created collection: {collection_name}")
        
        # Create index
        index_params = {
            "metric_type": METRIC_TYPE,
            "index_type": INDEX_TYPE,
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("✓ Created index")
        
        # Prepare test data - each field should be a list of values
        test_data = {
            "document_id": ["test_doc_001"],
            "page_number": [1],
            "chunk_id": [1],
            "text": ["This is a test document chunk."],
            "embedding": [[0.1] * DIMENSION]
        }
        
        # Verify data types
        print("Test data structure:")
        for key, value in test_data.items():
            print(f"  {key}: {type(value)} = {value}")
            if isinstance(value, list) and len(value) > 0:
                print(f"    First element type: {type(value[0])}")
                if key == "embedding":
                    print(f"    Embedding dimensions: {len(value[0])}")
        
        # Insert data
        print("\nInserting data...")
        collection.insert(test_data)
        collection.flush()
        print("✓ Successfully inserted test data")
        
        # Get stats
        stats = collection.get_statistics()
        print(f"✓ Collection stats: {stats}")
        
        # Clean up
        utility.drop_collection(collection_name)
        print("✓ Cleaned up test collection")
        
        connections.disconnect("default")
        print("✓ Disconnected from Milvus")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_insert()
    if success:
        print("\n✓ Test PASSED - Milvus insert is working!")
    else:
        print("\n✗ Test FAILED - Check the error above") 