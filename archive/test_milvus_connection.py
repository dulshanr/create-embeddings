#!/usr/bin/env python3
"""
Simple program to test Milvus connection.
"""

import os
from pymilvus import connections, utility
from dotenv import load_dotenv

def test_milvus_connection():
    """Test connection to Milvus server"""
    
    # Load environment variables
    load_dotenv()
    
    # Get connection settings
    host = os.getenv('MILVUS_HOST', 'localhost')
    port = os.getenv('MILVUS_PORT', '19530')
    user = os.getenv('MILVUS_USER', '')
    password = os.getenv('MILVUS_PASSWORD', '')
    
    print(f"Attempting to connect to Milvus at {host}:{port}")
    
    try:
        # Try to connect
        if user and password:
            connections.connect(
                alias="default",
                host=host,
                port=port,
                user=user,
                password=password
            )
            print("✓ Connected with authentication")
        else:
            connections.connect(
                alias="default",
                host=host,
                port=port
            )
            print("✓ Connected without authentication")
        
        # Test if we can list collections
        collections = utility.list_collections()
        print(f"✓ Successfully listed collections: {collections}")
        
        # Get server version info
        from pymilvus import __version__
        print(f"✓ PyMilvus version: {__version__}")
        
        # Disconnect
        connections.disconnect("default")
        print("✓ Disconnected successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def test_basic_operations():
    """Test basic Milvus operations"""
    
    print("\n--- Testing Basic Operations ---")
    
    try:
        # Connect
        connections.connect("default", host="localhost", port="19530")
        print("✓ Connected to Milvus")
        
        # List collections
        collections = utility.list_collections()
        print(f"✓ Found {len(collections)} collections: {collections}")
        
        # Test creating a simple collection (will be deleted)
        from pymilvus import Collection, CollectionSchema, FieldSchema, DataType
        
        # Define a simple schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
        ]
        schema = CollectionSchema(fields=fields, description="Test collection")
        
        # Create collection
        collection_name = "test_connection_collection"
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"✓ Dropped existing collection: {collection_name}")
        
        collection = Collection(name=collection_name, schema=schema)
        print(f"✓ Created test collection: {collection_name}")
        
        # Create index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        print("✓ Created index")
        
        # Insert some test data
        import numpy as np
        test_vectors = np.random.random([10, 128]).tolist()
        collection.insert([test_vectors])
        print("✓ Inserted test data")
        
        # Load collection
        collection.load()
        print("✓ Loaded collection")
        
        # Search
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[test_vectors[0]],
            anns_field="vector",
            param=search_params,
            limit=5
        )
        print("✓ Performed search")
        
        # Clean up
        utility.drop_collection(collection_name)
        print("✓ Cleaned up test collection")
        
        # Disconnect
        connections.disconnect("default")
        print("✓ Disconnected")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic operations failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Milvus Connection Test ===\n")
    
    # Test basic connection
    connection_success = test_milvus_connection()
    
    if connection_success:
        print("\n✓ Milvus connection test PASSED")
        
        # Test basic operations
        operations_success = test_basic_operations()
        
        if operations_success:
            print("\n✓ All tests PASSED - Milvus is working correctly!")
        else:
            print("\n✗ Basic operations test FAILED")
    else:
        print("\n✗ Milvus connection test FAILED")
        print("\nTroubleshooting tips:")
        print("1. Make sure Milvus is running")
        print("2. Check your connection settings in .env file")
        print("3. Verify host and port are correct")
        print("4. Check if authentication is required") 