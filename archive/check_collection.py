#!/usr/bin/env python3
"""
Check current collection schema and data.
"""

import os
from dotenv import load_dotenv
from pymilvus import connections, utility, Collection
from config import *

# Load environment variables
load_dotenv()

def check_collection():
    """Check the current collection"""
    
    print("=== Checking Current Collection ===\n")
    
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
        
        # Check if collection exists
        if utility.has_collection(collection_name):
            print(f"✓ Collection '{collection_name}' exists")
            
            # Get collection
            collection = Collection(collection_name)
            
            # Get schema
            schema = collection.schema
            print("\nCollection schema:")
            for field in schema.fields:
                print(f"  - {field.name}: {field.dtype} (max_length: {getattr(field, 'max_length', 'N/A')})")
            
            # Get statistics
            stats = collection.get_statistics()
            print(f"\nCollection statistics: {stats}")
            
            # List all collections
            all_collections = utility.list_collections()
            print(f"\nAll collections: {all_collections}")
            
        else:
            print(f"✗ Collection '{collection_name}' does not exist")
            
            # List all collections
            all_collections = utility.list_collections()
            print(f"Available collections: {all_collections}")
        
        connections.disconnect("default")
        print("✓ Disconnected from Milvus")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def clear_collection():
    """Clear the collection"""
    
    print("\n=== Clearing Collection ===\n")
    
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
        
        # Drop collection if it exists
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"✓ Dropped collection: {collection_name}")
        else:
            print(f"Collection '{collection_name}' does not exist")
        
        connections.disconnect("default")
        print("✓ Disconnected from Milvus")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Collection Check and Clear\n")
    
    # Check current collection
    check_success = check_collection()
    
    # Ask if user wants to clear
    print("\n" + "="*50)
    response = input("Do you want to clear the collection? (y/N): ").strip().lower()
    
    if response == 'y':
        clear_success = clear_collection()
        if clear_success:
            print("✓ Collection cleared successfully!")
        else:
            print("✗ Failed to clear collection")
    else:
        print("Collection not cleared")
    
    print("\nNext steps:")
    print("1. Run: python test_simple_insert.py")
    print("2. If that works, try your PDF processing again") 