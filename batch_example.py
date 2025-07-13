#!/usr/bin/env python3
"""
Example of using batch processing for large PDFs in Milvus.
"""

import os
import time
from pdf_to_milvus_batch import PDFToMilvusBatch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Example of batch processing large PDFs"""
    
    # Initialize the batch processing system
    logger.info("Initializing PDF to Milvus batch system...")
    
    # Configure batch sizes for large PDFs
    pdf_storage = PDFToMilvusBatch(
        batch_size=50,        # Process 50 chunks at a time
        embedding_batch_size=25  # Generate 25 embeddings at a time
    )
    
    if not pdf_storage.initialize():
        logger.error("Failed to initialize system")
        return
    
    try:
        # Example 1: Store a large PDF with progress tracking
        pdf_path = "./in/pdf_4.pdf"  # Replace with your large PDF path
        
        if os.path.exists(pdf_path):
            logger.info(f"Processing large PDF: {pdf_path}")
            start_time = time.time()
            
            success = pdf_storage.store_pdf_with_progress(pdf_path, document_id="large_doc_001")
            
            end_time = time.time()
            
            if success:
                logger.info(f"✓ Large PDF stored successfully! (took {end_time - start_time:.1f}s)")
            else:
                logger.error("✗ Failed to store large PDF")
        else:
            logger.warning(f"Large PDF file not found: {pdf_path}")
            logger.info("Please place a large PDF file named 'large_document.pdf' in the current directory")
        
        # Example 2: Store multiple large PDFs from a directory
        pdf_directory = "large_pdfs"  # Replace with your large PDF directory path
        
        if os.path.exists(pdf_directory):
            logger.info(f"Processing large PDFs from directory: {pdf_directory}")
            start_time = time.time()
            
            results = pdf_storage.store_multiple_pdfs_with_progress(pdf_directory)
            
            end_time = time.time()
            
            successful = sum(1 for result in results.values() if result["success"])
            total = len(results)
            total_processing_time = sum(result["processing_time"] for result in results.values())
            
            logger.info(f"✓ Successfully stored {successful}/{total} large PDFs")
            logger.info(f"  Total time: {end_time - start_time:.1f}s")
            logger.info(f"  Average time per PDF: {total_processing_time/total:.1f}s")
            
            # Show detailed results
            for pdf_file, result in results.items():
                status = "✓" if result["success"] else "✗"
                logger.info(f"  {status} {pdf_file}: {result['processing_time']:.1f}s")
        else:
            logger.info(f"Large PDF directory not found: {pdf_directory}")
            logger.info("Create a 'large_pdfs' directory and place your large PDF files there")
        
        # Example 3: Get detailed storage statistics
        stats = pdf_storage.get_storage_stats()
        logger.info("\nDetailed Storage Statistics:")
        logger.info(f"  Milvus Stats: {stats.get('milvus_stats', 'N/A')}")
        logger.info(f"  Embedding Dimension: {stats.get('embedding_dimension', 'N/A')}")
        logger.info(f"  Chunk Config: {stats.get('chunk_config', 'N/A')}")
        logger.info(f"  Batch Config: {stats.get('batch_config', 'N/A')}")
        
    finally:
        # Close the system
        pdf_storage.close()
        logger.info("PDF to Milvus batch system closed")

def demonstrate_batch_sizes():
    """Demonstrate different batch size configurations"""
    
    logger.info("Demonstrating different batch size configurations...")
    
    # Configuration for different scenarios
    configurations = [
        {
            "name": "Small batches (memory-constrained)",
            "batch_size": 25,
            "embedding_batch_size": 10
        },
        {
            "name": "Medium batches (balanced)",
            "batch_size": 50,
            "embedding_batch_size": 25
        },
        {
            "name": "Large batches (high-performance)",
            "batch_size": 100,
            "embedding_batch_size": 50
        }
    ]
    
    for config in configurations:
        logger.info(f"\n--- {config['name']} ---")
        logger.info(f"Batch size: {config['batch_size']}")
        logger.info(f"Embedding batch size: {config['embedding_batch_size']}")
        
        # Initialize with this configuration
        pdf_storage = PDFToMilvusBatch(
            batch_size=config['batch_size'],
            embedding_batch_size=config['embedding_batch_size']
        )
        
        if pdf_storage.initialize():
            stats = pdf_storage.get_storage_stats()
            logger.info(f"System initialized successfully")
            logger.info(f"Batch config: {stats.get('batch_config', 'N/A')}")
            pdf_storage.close()
        else:
            logger.error("Failed to initialize system")

if __name__ == "__main__":
    # Run the main example
    main()
    
    # Uncomment to demonstrate different batch sizes
    # demonstrate_batch_sizes() 