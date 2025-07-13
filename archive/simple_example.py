#!/usr/bin/env python3
"""
Simple example of storing PDFs in Milvus without retrieval/generation.
"""

import os
from pdf_to_milvus import PDFToMilvus
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Example of storing PDFs in Milvus"""
    
    # Initialize the PDF storage system
    logger.info("Initializing PDF to Milvus system...")
    pdf_storage = PDFToMilvus()
    
    if not pdf_storage.initialize():
        logger.error("Failed to initialize system")
        return
    
    try:
        # Example 1: Store a single PDF
        pdf_path = "sample_document.pdf"  # Replace with your PDF path
        
        if os.path.exists(pdf_path):
            logger.info(f"Storing PDF: {pdf_path}")
            success = pdf_storage.store_pdf(pdf_path, document_id="sample_doc_001")
            
            if success:
                logger.info("✓ PDF stored successfully!")
            else:
                logger.error("✗ Failed to store PDF")
        else:
            logger.warning(f"PDF file not found: {pdf_path}")
            logger.info("Please place a PDF file named 'sample_document.pdf' in the current directory")
        
        # Example 2: Store multiple PDFs from a directory
        pdf_directory = "pdfs"  # Replace with your PDF directory path
        
        if os.path.exists(pdf_directory):
            logger.info(f"Storing PDFs from directory: {pdf_directory}")
            results = pdf_storage.store_multiple_pdfs(pdf_directory)
            
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            logger.info(f"✓ Successfully stored {successful}/{total} PDFs")
        else:
            logger.info(f"PDF directory not found: {pdf_directory}")
            logger.info("Create a 'pdfs' directory and place your PDF files there")
        
        # Example 3: Get storage statistics
        stats = pdf_storage.get_storage_stats()
        logger.info("\nStorage Statistics:")
        logger.info(f"  Milvus Stats: {stats.get('milvus_stats', 'N/A')}")
        logger.info(f"  Embedding Dimension: {stats.get('embedding_dimension', 'N/A')}")
        logger.info(f"  Chunk Config: {stats.get('chunk_config', 'N/A')}")
        
    finally:
        # Close the system
        pdf_storage.close()
        logger.info("PDF to Milvus system closed")

if __name__ == "__main__":
    main() 