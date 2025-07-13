#!/usr/bin/env python3
"""
Script to drop the Milvus collection.
"""

import argparse
import logging
from src.milvus_client import MilvusClient
from config import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def drop_collection(force: bool = False):
    """Drop the Milvus collection"""
    try:
        client = MilvusClient()
        
        # Connect to Milvus
        if not client.connect():
            logger.error("Failed to connect to Milvus")
            return False
        
        # Get collection stats before dropping
        stats = client.get_collection_stats()
        if "error" not in stats:
            logger.info(f"Collection '{COLLECTION_NAME}' currently has {stats.get('row_count', 0)} records")
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' may not exist or is not accessible")
        
        # Confirm deletion unless forced
        if not force:
            confirm = input(f"\nAre you sure you want to drop collection '{COLLECTION_NAME}'? (y/N): ").strip().lower()
            if confirm != 'y':
                logger.info("Operation cancelled by user")
                return False
        
        # Drop the collection
        success = client.delete_collection()
        
        if success:
            logger.info(f"✓ Successfully dropped collection: {COLLECTION_NAME}")
            return True
        else:
            logger.error(f"✗ Failed to drop collection: {COLLECTION_NAME}")
            return False
            
    except Exception as e:
        logger.error(f"Error dropping collection: {e}")
        return False
    finally:
        try:
            client.close()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Drop the Milvus collection")
    parser.add_argument("--force", action="store_true", help="Force deletion without confirmation")
    
    args = parser.parse_args()
    
    logger.info(f"Attempting to drop collection: {COLLECTION_NAME}")
    
    success = drop_collection(force=args.force)
    
    if success:
        logger.info("Collection dropped successfully!")
    else:
        logger.error("Failed to drop collection")
        exit(1)

if __name__ == "__main__":
    main() 