#!/usr/bin/env python3
"""
Basic Milvus insert test using the simplest format.
"""

import os
from dotenv import load_dotenv
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from config import *

# Load environment variables
load_dotenv()

def test_basic_insert():
    """Test basic insert with minimal data"""
    
    print("=== Basic Milvus Insert Test ===\n")
    
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
        collection_name = "test_basic_collection"
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"✓ Dropped existing collection: {collection_name}")
        
        # Create a simple collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="value", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(fields=fields, description="Basic test collection")
        collection = Collection(name=collection_name, schema=schema)
        print(f"✓ Created collection: {collection_name}")
        
        # Test 1: Insert with dictionary format
        print("\n--- Test 1: Dictionary Format ---")
        try:
            data1 = {
                "name": ["test1", "test2"],
                "value": [1, 2]
            }
            print(f"Data: {data1}")
            collection.insert(data1)
            collection.flush()
            print("✓ Test 1: Success")
        except Exception as e:
            print(f"✗ Test 1 failed: {e}")
        
        # Test 2: Insert with list of dictionaries
        print("\n--- Test 2: List of Dictionaries ---")
        try:
            # Clear collection
            utility.drop_collection(collection_name)
            collection = Collection(name=collection_name, schema=schema)
            
            data2 = [
                {"name": "test1", "value": 1},
                {"name": "test2", "value": 2}
            ]
            print(f"Data: {data2}")
            collection.insert(data2)
            collection.flush()
            print("✓ Test 2: Success")
        except Exception as e:
            print(f"✗ Test 2 failed: {e}")
        
        # Test 3: Insert with list of lists
        print("\n--- Test 3: List of Lists ---")
        try:
            # Clear collection
            utility.drop_collection(collection_name)
            collection = Collection(name=collection_name, schema=schema)
            
            data3 = [
                ["test1", 1],
                ["test2", 2]
            ]
            print(f"Data: {data3}")
            collection.insert(data3)
            collection.flush()
            print("✓ Test 3: Success")
        except Exception as e:
            print(f"✗ Test 3 failed: {e}")
        
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
    success = test_basic_insert()
    if success:
        print("\n✓ Basic test completed!")
    else:
        print("\n✗ Basic test failed!") 