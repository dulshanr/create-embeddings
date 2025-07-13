#!/usr/bin/env python3
"""
Simple PDF to Milvus storage system.
Only handles PDF processing and vector storage, no retrieval/generation.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
import logging
from milvus_client import MilvusClient
from pdf_processor import PDFProcessor
from embedding_generator import EmbeddingGenerator
from config import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFToMilvus:
    def __init__(self):
        """Initialize the PDF to Milvus storage system"""
        self.milvus_client = MilvusClient()
        self.pdf_processor = PDFProcessor()
        self.embedding_generator = EmbeddingGenerator()
        
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
            
            logger.info("PDF to Milvus system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def store_pdf(self, pdf_path: str, document_id: Optional[str] = None) -> bool:
        """Store a PDF document in Milvus"""
        try:
            # Validate PDF file
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            # Generate document ID if not provided
            if not document_id:
                document_id = str(uuid.uuid4())
            
            logger.info(f"Processing PDF: {pdf_path} with document ID: {document_id}")
            
            # Process PDF into chunks
            chunks = self.pdf_processor.process_pdf(pdf_path, document_id)
            
            if not chunks:
                logger.error(f"No chunks generated from PDF: {pdf_path}")
                return False
            
            # Generate embeddings for chunks
            chunks_with_embeddings = self.embedding_generator.add_embeddings_to_chunks(chunks)
            
            if not chunks_with_embeddings:
                logger.error("Failed to generate embeddings for chunks")
                return False
            
            # Validate embeddings
            valid_chunks = []
            for chunk in chunks_with_embeddings:
                if self.embedding_generator.validate_embedding(chunk.get("embedding", [])):
                    valid_chunks.append(chunk)
                else:
                    logger.warning(f"Invalid embedding for chunk {chunk.get('chunk_id')}")
            
            if not valid_chunks:
                logger.error("No valid chunks with embeddings")
                return False
            
            # Store in Milvus
            if not self.milvus_client.insert_documents(valid_chunks):
                logger.error("Failed to insert documents into Milvus")
                return False
            
            logger.info(f"Successfully stored PDF with {len(valid_chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store PDF {pdf_path}: {e}")
            return False
    
    def store_multiple_pdfs(self, pdf_directory: str) -> Dict[str, bool]:
        """Store multiple PDFs from a directory"""
        results = {}
        
        try:
            if not os.path.exists(pdf_directory):
                logger.error(f"Directory not found: {pdf_directory}")
                return results
            
            pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                logger.warning(f"No PDF files found in directory: {pdf_directory}")
                return results
            
            logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_directory, pdf_file)
                document_id = os.path.splitext(pdf_file)[0]  # Use filename as document ID
                
                success = self.store_pdf(pdf_path, document_id)
                results[pdf_file] = success
                
                if success:
                    logger.info(f"✓ Successfully stored: {pdf_file}")
                else:
                    logger.error(f"✗ Failed to store: {pdf_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process PDF directory: {e}")
            return results
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        try:
            stats = {
                "milvus_stats": self.milvus_client.get_collection_stats(),
                "embedding_dimension": self.embedding_generator.get_embedding_dimension(),
                "chunk_config": {
                    "chunk_size": self.pdf_processor.chunk_size,
                    "chunk_overlap": self.pdf_processor.chunk_overlap
                }
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
            logger.info("PDF to Milvus system closed")
        except Exception as e:
            logger.error(f"Failed to close system: {e}")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Store PDF documents in Milvus")
    parser.add_argument("--pdf", help="Path to a single PDF file")
    parser.add_argument("--directory", help="Path to directory containing PDF files")
    parser.add_argument("--clear", action="store_true", help="Clear all documents from Milvus")
    parser.add_argument("--stats", action="store_true", help="Show storage statistics")
    
    args = parser.parse_args()
    
    # Initialize system
    pdf_storage = PDFToMilvus()
    
    if not pdf_storage.initialize():
        logger.error("Failed to initialize system")
        return
    
    try:
        if args.clear:
            success = pdf_storage.clear_all_documents()
            if success:
                logger.info("✓ All documents cleared from Milvus")
            else:
                logger.error("✗ Failed to clear documents")
        
        elif args.stats:
            stats = pdf_storage.get_storage_stats()
            logger.info("Storage Statistics:")
            logger.info(f"  Milvus Stats: {stats.get('milvus_stats', 'N/A')}")
            logger.info(f"  Embedding Dimension: {stats.get('embedding_dimension', 'N/A')}")
            logger.info(f"  Chunk Config: {stats.get('chunk_config', 'N/A')}")
        
        elif args.pdf:
            success = pdf_storage.store_pdf(args.pdf)
            if success:
                logger.info("✓ PDF stored successfully!")
            else:
                logger.error("✗ Failed to store PDF")
        
        elif args.directory:
            results = pdf_storage.store_multiple_pdfs(args.directory)
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            logger.info(f"✓ Successfully stored {successful}/{total} PDFs")
        
        else:
            logger.info("No action specified. Use --help for usage information.")
    
    finally:
        pdf_storage.close()

if __name__ == "__main__":
    main() 