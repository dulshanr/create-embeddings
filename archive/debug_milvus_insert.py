#!/usr/bin/env python3
"""
Debug script to test Milvus insert functionality.
"""

import os
import numpy as np
from milvus_client import MilvusClient
from config import *

def test_insert_single_document():
    """Test inserting a single document with proper data types"""
    
    print("=== Testing Milvus Insert with Single Document ===\n")
    
    # Initialize client
    client = MilvusClient()
    
    if not client.connect():
        print("✗ Failed to connect to Milvus")
        return False
    
    if not client.create_collection():
        print("✗ Failed to create collection")
        return False
    
    # Create test document with proper data types
    test_document = {
        "document_id": "test_doc_001",
        "page_number": 1,
        "chunk_id": 1,
        "text": "This is a test document chunk for debugging purposes.",
        "embedding": [0.1] * DIMENSION  # Create a dummy embedding
    }
    
    print(f"Test document data types:")
    print(f"  document_id: {type(test_document['document_id'])} = {test_document['document_id']}")
    print(f"  page_number: {type(test_document['page_number'])} = {test_document['page_number']}")
    print(f"  chunk_id: {type(test_document['chunk_id'])} = {test_document['chunk_id']}")
    print(f"  text: {type(test_document['text'])} = {test_document['text'][:50]}...")
    print(f"  embedding: {type(test_document['embedding'])} = {len(test_document['embedding'])} dimensions")
    
    # Test insert
    try:
        success = client.insert_documents([test_document])
        if success:
            print("✓ Successfully inserted test document")
            
            # Get stats
            stats = client.get_collection_stats()
            print(f"✓ Collection stats: {stats}")
            
            return True
        else:
            print("✗ Failed to insert test document")
            return False
    except Exception as e:
        print(f"✗ Exception during insert: {e}")
        return False
    finally:
        client.close()

def test_insert_multiple_documents():
    """Test inserting multiple documents"""
    
    print("\n=== Testing Milvus Insert with Multiple Documents ===\n")
    
    # Initialize client
    client = MilvusClient()
    
    if not client.connect():
        print("✗ Failed to connect to Milvus")
        return False
    
    if not client.create_collection():
        print("✗ Failed to create collection")
        return False
    
    # Create test documents
    test_documents = []
    for i in range(3):
        doc = {
            "document_id": f"test_doc_{i+1:03d}",
            "page_number": i + 1,
            "chunk_id": i + 1,
            "text": f"This is test document chunk {i+1} for debugging purposes.",
            "embedding": [0.1 + i * 0.01] * DIMENSION  # Different embeddings
        }
        test_documents.append(doc)
    
    print(f"Created {len(test_documents)} test documents")
    
    # Test insert
    try:
        success = client.insert_documents(test_documents)
        if success:
            print("✓ Successfully inserted multiple test documents")
            
            # Get stats
            stats = client.get_collection_stats()
            print(f"✓ Collection stats: {stats}")
            
            return True
        else:
            print("✗ Failed to insert multiple test documents")
            return False
    except Exception as e:
        print(f"✗ Exception during insert: {e}")
        return False
    finally:
        client.close()

def test_collection_schema():
    """Test and display collection schema"""
    
    print("\n=== Testing Collection Schema ===\n")
    
    # Initialize client
    client = MilvusClient()
    
    if not client.connect():
        print("✗ Failed to connect to Milvus")
        return False
    
    if not client.create_collection():
        print("✗ Failed to create collection")
        return False
    
    try:
        # Get collection schema
        collection = client.collection
        schema = collection.schema
        
        print("Collection schema:")
        for field in schema.fields:
            print(f"  - {field.name}: {field.dtype} (max_length: {getattr(field, 'max_length', 'N/A')})")
        
        return True
    except Exception as e:
        print(f"✗ Exception getting schema: {e}")
        return False
    finally:
        client.close()

if __name__ == "__main__":
    print("Milvus Insert Debug Test\n")
    
    # Test collection schema
    schema_success = test_collection_schema()
    
    # Test single document insert
    single_success = test_insert_single_document()
    
    # Test multiple documents insert
    multiple_success = test_insert_multiple_documents()
    
    print("\n=== Summary ===")
    print(f"Schema test: {'✓ PASSED' if schema_success else '✗ FAILED'}")
    print(f"Single insert: {'✓ PASSED' if single_success else '✗ FAILED'}")
    print(f"Multiple insert: {'✓ PASSED' if multiple_success else '✗ FAILED'}")
    
    if all([schema_success, single_success, multiple_success]):
        print("\n✓ All tests PASSED - Milvus insert is working correctly!")
    else:
        print("\n✗ Some tests FAILED - Check the errors above") 