#!/usr/bin/env python3
"""
Ultra Memory Optimized PDF to Milvus Processor
Designed for processing very large PDFs (500+ pages) with minimal memory usage
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pdf_processor_memory_optimized import UltraMemoryOptimizedPDFProcessor
from src.milvus_client import MilvusClient
from src.embedding_generator import EmbeddingGenerator
from ultra_optimized_config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class UltraOptimizedPDFProcessor:
    def __init__(self):
        self.pdf_processor = UltraMemoryOptimizedPDFProcessor()
        self.milvus_client = MilvusClient()
        self.embedding_generator = EmbeddingGenerator()
        
        # Ultra-conservative settings for large PDFs
        self.pdf_processor.max_pages_per_batch = ULTRA_PAGES_PER_BATCH
        self.pdf_processor.max_memory_mb = ULTRA_MEMORY_MB
        self.batch_size = ULTRA_BATCH_SIZE
        
    def process_pdf_ultra_safe(self, pdf_path: str, document_id: str = None) -> bool:
        """Process PDF with ultra-conservative memory management"""
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            if document_id is None:
                document_id = Path(pdf_path).stem
            
            # Check file size
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            logger.info(f"Processing PDF: {pdf_path}")
            logger.info(f"File size: {file_size_mb:.2f} MB")
            logger.info(f"Document ID: {document_id}")
            
            # Get initial memory usage
            initial_memory = self.pdf_processor._get_memory_usage_mb()
            logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
            
            # Process PDF with memory monitoring
            total_chunks = 0
            total_embeddings = 0
            
            start_time = time.time()
            
            for batch in self.pdf_processor.process_pdf_with_memory_monitoring(
                pdf_path, document_id, max_memory_mb=ULTRA_MEMORY_MB
            ):
                try:
                    # Generate embeddings for this batch
                    embeddings_batch = []
                    
                    for chunk in batch:
                        # Generate embedding
                        embedding = self.embedding_generator.generate_single_embedding(chunk["text"])
                        
                        if embedding is not None:
                            # Prepare data for Milvus
                            data = {
                                "document_id": chunk["document_id"],
                                "page_number": chunk["page_number"],
                                "chunk_id": chunk["chunk_id"],
                                "text": chunk["text"],
                                "embedding": embedding,
                                "page_width": chunk.get("page_width", 0),
                                "page_height": chunk.get("page_height", 0)
                            }
                            embeddings_batch.append(data)
                    
                    # Insert batch into Milvus
                    if embeddings_batch:
                        success = self.milvus_client.insert_data(embeddings_batch)
                        if success:
                            total_embeddings += len(embeddings_batch)
                            logger.info(f"Inserted {len(embeddings_batch)} embeddings. Total: {total_embeddings}")
                        else:
                            logger.error(f"Failed to insert batch of {len(embeddings_batch)} embeddings")
                    
                    total_chunks += len(batch)
                    
                    # Log progress
                    current_memory = self.pdf_processor._get_memory_usage_mb()
                    elapsed_time = time.time() - start_time
                    logger.info(f"Progress: {total_chunks} chunks, {total_embeddings} embeddings. "
                              f"Memory: {current_memory:.2f} MB. Time: {elapsed_time:.1f}s")
                    
                    # Force memory cleanup after each batch
                    self.pdf_processor._force_memory_cleanup()
                    
                    # Small delay to allow system to stabilize
                    time.sleep(BATCH_PROCESSING_DELAY)
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    continue
            
            # Final cleanup
            self.pdf_processor._force_memory_cleanup()
            
            final_memory = self.pdf_processor._get_memory_usage_mb()
            total_time = time.time() - start_time
            
            logger.info(f"Processing completed!")
            logger.info(f"Total chunks processed: {total_chunks}")
            logger.info(f"Total embeddings inserted: {total_embeddings}")
            logger.info(f"Final memory usage: {final_memory:.2f} MB")
            logger.info(f"Total processing time: {total_time:.1f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            return False
    
    def process_multiple_pdfs(self, pdf_directory: str) -> Dict[str, bool]:
        """Process multiple PDFs in a directory"""
        results = {}
        
        try:
            pdf_dir = Path(pdf_directory)
            if not pdf_dir.exists():
                logger.error(f"Directory not found: {pdf_directory}")
                return results
            
            pdf_files = list(pdf_dir.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            for i, pdf_file in enumerate(pdf_files, 1):
                logger.info(f"Processing PDF {i}/{len(pdf_files)}: {pdf_file.name}")
                
                success = self.process_pdf_ultra_safe(str(pdf_file))
                results[pdf_file.name] = success
                
                if success:
                    logger.info(f"Successfully processed: {pdf_file.name}")
                else:
                    logger.error(f"Failed to process: {pdf_file.name}")
                
                # Wait between files to allow system to stabilize
                time.sleep(FILE_PROCESSING_DELAY)
            
            # Summary
            successful = sum(1 for success in results.values() if success)
            logger.info(f"Processing summary: {successful}/{len(pdf_files)} PDFs processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing multiple PDFs: {e}")
        
        return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra Memory Optimized PDF to Milvus Processor")
    parser.add_argument("--pdf", type=str, help="Path to single PDF file")
    parser.add_argument("--directory", type=str, help="Path to directory containing PDFs")
    parser.add_argument("--document-id", type=str, help="Custom document ID (for single PDF)")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    processor = UltraOptimizedPDFProcessor()
    
    if args.pdf:
        # Process single PDF
        success = processor.process_pdf_ultra_safe(args.pdf, args.document_id)
        if success:
            logger.info("PDF processing completed successfully!")
        else:
            logger.error("PDF processing failed!")
            sys.exit(1)
    
    elif args.directory:
        # Process multiple PDFs
        results = processor.process_multiple_pdfs(args.directory)
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        if successful == total:
            logger.info("All PDFs processed successfully!")
        else:
            logger.warning(f"Only {successful}/{total} PDFs processed successfully")
            sys.exit(1)
    
    else:
        # Default: process PDFs in the 'in' directory
        if os.path.exists("in"):
            results = processor.process_multiple_pdfs("in")
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            
            if successful == total:
                logger.info("All PDFs processed successfully!")
            else:
                logger.warning(f"Only {successful}/{total} PDFs processed successfully")
                sys.exit(1)
        else:
            logger.error("No PDF file or directory specified, and 'in' directory not found")
            parser.print_help()
            sys.exit(1)

if __name__ == "__main__":
    main() 