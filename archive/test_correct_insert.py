#!/usr/bin/env python3
"""
Test with correct Milvus insert format.
"""

import os
from dotenv import load_dotenv
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from config import *

# Load environment variables
load_dotenv()

def test_correct_insert():
    """Test insert using the correct Milvus data format"""
    
    print("=== Correct Milvus Insert Test ===\n")
    
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
        
        # Test Method 1: Insert with data as separate lists (current method)
        print("\n--- Testing Method 1: Separate Lists ---")
        try:
            data_method1 = {
                "document_id": ["test_doc_001", "test_doc_001"],
                "page_number": [1, 1],
                "chunk_id": [1, 2],
                "text": ["This is test chunk 1.", "This is test chunk 2."],
                "embedding": [[0.1] * DIMENSION, [0.2] * DIMENSION]
            }
            
            print("Data structure for Method 1:")
            for key, value in data_method1.items():
                print(f"  {key}: {type(value)} = {value}")
            
            collection.insert(data_method1)
            collection.flush()
            print("✓ Method 1: Successfully inserted data")
            
        except Exception as e:
            print(f"✗ Method 1 failed: {e}")
        
        # Test Method 2: Insert with data as rows
        print("\n--- Testing Method 2: Data as Rows ---")
        try:
            # Clear collection first
            utility.drop_collection(collection_name)
            collection = Collection(name=collection_name, schema=schema)
            collection.create_index(field_name="embedding", index_params=index_params)
            
            # Insert data as rows
            rows = [
                ["test_doc_001", 1, 1, "This is test chunk 1.", [0.1] * DIMENSION],
                ["test_doc_001", 1, 2, "This is test chunk 2.", [0.2] * DIMENSION]
            ]
            
            print("Data structure for Method 2:")
            print(f"  Rows: {type(rows)} = {rows}")
            
            collection.insert(rows)
            collection.flush()
            print("✓ Method 2: Successfully inserted data")
            
        except Exception as e:
            print(f"✗ Method 2 failed: {e}")
        
        # Test Method 3: Insert with explicit field names
        print("\n--- Testing Method 3: Explicit Field Names ---")
        try:
            # Clear collection first
            utility.drop_collection(collection_name)
            collection = Collection(name=collection_name, schema=schema)
            collection.create_index(field_name="embedding", index_params=index_params)
            
            # Insert with explicit field names
            data_method3 = [
                {
                    "document_id": "test_doc_001",
                    "page_number": 1,
                    "chunk_id": 1,
                    "text": "This is test chunk 1.",
                    "embedding": [0.1] * DIMENSION
                },
                {
                    "document_id": "test_doc_001",
                    "page_number": 1,
                    "chunk_id": 2,
                    "text": "This is test chunk 2.",
                    "embedding": [0.2] * DIMENSION
                }
            ]
            
            print("Data structure for Method 3:")
            print(f"  Data: {type(data_method3)} = {data_method3}")
            
            collection.insert(data_method3)
            collection.flush()
            print("✓ Method 3: Successfully inserted data")
            
        except Exception as e:
            print(f"✗ Method 3 failed: {e}")
        
        # Get final stats
        stats = collection.get_statistics()
        print(f"\n✓ Final collection stats: {stats}")
        
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
    success = test_correct_insert()
    if success:
        print("\n✓ Test completed!")
    else:
        print("\n✗ Test failed!") 