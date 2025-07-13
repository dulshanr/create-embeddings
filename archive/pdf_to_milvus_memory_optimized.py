#!/usr/bin/env python3
"""
Memory-optimized PDF to Milvus storage system for large PDFs.
Processes pages one at a time to prevent memory crashes.
"""

import os
import uuid
import time
import gc
from typing import List, Dict, Any, Optional, Generator
import logging
from tqdm import tqdm
from src.milvus_client import MilvusClient
from src.pdf_processor_memory_optimized import MemoryOptimizedPDFProcessor
from src.embedding_generator import EmbeddingGenerator, OpenAIEmbeddingGenerator
from config import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryOptimizedPDFToMilvus:
    def __init__(self, batch_size: int = 50, embedding_batch_size: int = 25):
        """Initialize the memory-optimized PDF to Milvus system"""
        self.milvus_client = MilvusClient()
        self.pdf_processor = MemoryOptimizedPDFProcessor()
        
        # Choose embedding generator based on configuration
        if USE_OPENAI_EMBEDDINGS:
            self.embedding_generator = OpenAIEmbeddingGenerator(OPENAI_MODEL)
            logger.info(f"Using OpenAI embeddings with model: {OPENAI_MODEL}")
        else:
            self.embedding_generator = EmbeddingGenerator()
            logger.info(f"Using sentence-transformers with model: {EMBEDDING_MODEL}")
        
        self.batch_size = batch_size  # Number of chunks to process at once
        self.embedding_batch_size = embedding_batch_size  # Number of texts to embed at once
        
    def initialize(self) -> bool:
        """Initialize the system by connecting to Milvus and creating collection"""
        try:
            # Connect to Milvus
            if not self.milvus_client.connect():
                logger.error("Failed to connect to Milvus")
                return False
            
            # Create collection
            if not self.milvus_client.create_collection():
                logger.error("Failed to create collection")
                return False
            
            logger.info("Memory-optimized PDF to Milvus system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def generate_embeddings_in_batches(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks in smaller batches"""
        try:
            chunks_with_embeddings = []
            
            # Process embedding generation in smaller batches
            for i in range(0, len(chunks), self.embedding_batch_size):
                batch = chunks[i:i + self.embedding_batch_size]
                texts = [chunk["text"] for chunk in batch]
                
                # Generate embeddings for this batch
                embeddings = self.embedding_generator.generate_embeddings(texts)
                
                # Add embeddings to chunks
                for j, chunk in enumerate(batch):
                    if j < len(embeddings):
                        chunk["embedding"] = embeddings[j]
                        chunks_with_embeddings.append(chunk)
                    else:
                        logger.warning(f"Missing embedding for chunk {chunk.get('chunk_id')}")
                
                # Force garbage collection after each embedding batch
                gc.collect()
            
            return chunks_with_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []
    
    def store_pdf_memory_optimized(self, pdf_path: str, document_id: Optional[str] = None) -> bool:
        """Store a PDF document in Milvus with memory optimization"""
        try:
            # Validate PDF file
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            # Generate document ID if not provided
            if not document_id:
                document_id = str(uuid.uuid4())
            
            logger.info(f"Processing PDF: {pdf_path} with document ID: {document_id}")
            
            # Get initial memory usage
            memory_info = self.pdf_processor.get_memory_usage_info()
            if memory_info:
                logger.info(f"Initial memory usage: {memory_info.get('rss_mb', 0):.1f} MB")
            
            total_chunks_processed = 0
            total_chunks_stored = 0
            failed_chunks = 0
            
            # Process PDF page by page
            with tqdm(desc="Processing pages", unit="page") as pbar:
                for page_chunks in self.pdf_processor.process_pdf_generator(pdf_path, document_id):
                    if not page_chunks:
                        continue
                    
                    # Generate embeddings for this page's chunks
                    chunks_with_embeddings = self.generate_embeddings_in_batches(page_chunks)
                    
                    if not chunks_with_embeddings:
                        logger.warning(f"Failed to generate embeddings for page chunks")
                        failed_chunks += len(page_chunks)
                        pbar.update(1)
                        continue
                    
                    # Validate embeddings
                    valid_chunks = []
                    for chunk in chunks_with_embeddings:
                        if self.embedding_generator.validate_embedding(chunk.get("embedding", [])):
                            valid_chunks.append(chunk)
                        else:
                            logger.warning(f"Invalid embedding for chunk {chunk.get('chunk_id')}")
                    
                    if not valid_chunks:
                        logger.warning(f"No valid chunks in page")
                        failed_chunks += len(page_chunks)
                        pbar.update(1)
                        continue
                    
                    # Store chunks in Milvus
                    if self.milvus_client.insert_documents(valid_chunks):
                        total_chunks_stored += len(valid_chunks)
                        logger.debug(f"Successfully stored {len(valid_chunks)} chunks from page")
                    else:
                        logger.error(f"Failed to store chunks from page")
                        failed_chunks += len(valid_chunks)
                    
                    total_chunks_processed += len(page_chunks)
                    pbar.update(1)
                    
                    # Force garbage collection after each page
                    gc.collect()
                    
                    # Log memory usage every 50 pages
                    if total_chunks_processed % 50 == 0:
                        memory_info = self.pdf_processor.get_memory_usage_info()
                        if memory_info:
                            logger.info(f"Memory usage: {memory_info.get('rss_mb', 0):.1f} MB")
            
            # Final memory usage
            memory_info = self.pdf_processor.get_memory_usage_info()
            if memory_info:
                logger.info(f"Final memory usage: {memory_info.get('rss_mb', 0):.1f} MB")
            
            logger.info(f"PDF processing completed:")
            logger.info(f"  ✓ Successfully stored: {total_chunks_stored} chunks")
            logger.info(f"  ✗ Failed: {failed_chunks} chunks")
            logger.info(f"  📊 Success rate: {total_chunks_stored/(total_chunks_stored+failed_chunks)*100:.1f}%")
            
            return total_chunks_stored > 0
            
        except Exception as e:
            logger.error(f"Failed to store PDF {pdf_path}: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            stats = {
                "milvus_stats": self.milvus_client.get_collection_stats(),
                "embedding_dimension": self.embedding_generator.get_embedding_dimension(),
                "chunk_config": {
                    "chunk_size": self.pdf_processor.chunk_size,
                    "chunk_overlap": self.pdf_processor.chunk_overlap
                },
                "memory_usage": self.pdf_processor.get_memory_usage_info()
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}
    
    def clear_all_documents(self) -> bool:
        """Clear all documents from Milvus"""
        try:
            return self.milvus_client.delete_collection()
        except Exception as e:
            logger.error(f"Failed to clear all documents: {e}")
            return False
    
    def close(self):
        """Close the system"""
        try:
            self.milvus_client.close()
            logger.info("Memory-optimized PDF to Milvus system closed")
        except Exception as e:
            logger.error(f"Failed to close system: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Store PDF documents in Milvus with memory optimization")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--document-id", help="Document ID (optional)")
    parser.add_argument("--clear", action="store_true", help="Clear all documents from Milvus")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--embedding-batch-size", type=int, default=25, help="Batch size for embeddings")
    
    args = parser.parse_args()
    
    # Initialize system
    pdf_storage = MemoryOptimizedPDFToMilvus(
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size
    )
    
    if not pdf_storage.initialize():
        logger.error("Failed to initialize system")
        return
    
    try:
        if args.clear:
            logger.info("Clearing all documents from Milvus...")
            if pdf_storage.clear_all_documents():
                logger.info("✓ All documents cleared from Milvus")
            else:
                logger.error("✗ Failed to clear documents")
            return
        
        # Store PDF
        success = pdf_storage.store_pdf_memory_optimized(args.pdf_path, args.document_id)
        
        if success:
            logger.info("✓ PDF stored successfully!")
            
            # Get and display stats
            stats = pdf_storage.get_storage_stats()
            logger.info(f"  Milvus Stats: {stats.get('milvus_stats', 'N/A')}")
            logger.info(f"  Memory Usage: {stats.get('memory_usage', 'N/A')}")
        else:
            logger.error("✗ Failed to store PDF")
    
    finally:
        pdf_storage.close()

if __name__ == "__main__":
    main() 