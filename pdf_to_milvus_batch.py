#!/usr/bin/env python3
"""
PDF to Milvus storage system with batch processing for large PDFs.
Handles memory-efficient processing of large documents.
"""

import os
import uuid
import time
from typing import List, Dict, Any, Optional, Generator
import logging
from tqdm import tqdm
from src.milvus_client import MilvusClient
from src.pdf_processor import PDFProcessor
from src.embedding_generator import EmbeddingGenerator, OpenAIEmbeddingGenerator
# from text_validation import validate_pdf_processing
from config import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFToMilvusBatch:
    def __init__(self, batch_size: int = 100, embedding_batch_size: int = 50):
        """Initialize the batch processing PDF to Milvus system"""
        self.milvus_client = MilvusClient()
        self.pdf_processor = PDFProcessor()
        
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
            
            logger.info("PDF to Milvus batch system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def process_pdf_in_batches(self, pdf_path: str, document_id: Optional[str] = None) -> Generator[List[Dict[str, Any]], None, None]:
        """Process PDF and yield chunks in batches"""
        try:
            # Validate PDF file
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return
            
            # Generate document ID if not provided
            if not document_id:
                document_id = str(uuid.uuid4())
            
            logger.info(f"Processing PDF: {pdf_path} with document ID: {document_id}")
            
            # Process PDF into chunks
            all_chunks = self.pdf_processor.process_pdf(pdf_path, document_id)
            
            if not all_chunks:
                logger.error(f"No chunks generated from PDF: {pdf_path}")
                return
            
            logger.info(f"Generated {len(all_chunks)} total chunks, processing in batches of {self.batch_size}")
            
            # Yield chunks in batches
            for i in range(0, len(all_chunks), self.batch_size):
                batch = all_chunks[i:i + self.batch_size]
                yield batch
                
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            return
    
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
            
            # Write chunks_with_embeddings to file
            try:
                with open("logs/chunks_with_embeddings_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n=== Batch of {len(chunks_with_embeddings)} chunks with embeddings ===\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    for i, chunk in enumerate(chunks_with_embeddings):
                        f.write(f"\nChunk {i+1}:\n")
                        f.write(f"  Document ID: {chunk.get('document_id', 'N/A')}\n")
                        f.write(f"  Page Number: {chunk.get('page_number', 'N/A')}\n")
                        f.write(f"  Chunk ID: {chunk.get('chunk_id', 'N/A')}\n")
                        f.write(f"  Text: {chunk.get('text', 'N/A')}...\n")  # First 200 chars
                        f.write(f"  Embedding length: {len(chunk.get('embedding', []))}\n")
                        f.write(f"  Embedding preview: {chunk.get('embedding', [])[:5]}...\n")  # First 5 values
                    f.write("-" * 80 + "\n")
            except Exception as e:
                logger.warning(f"Failed to write chunks_with_embeddings to log file: {e}")
            
            return chunks_with_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []
    
    def store_pdf_with_progress(self, pdf_path: str, document_id: Optional[str] = None) -> bool:
        """Store a PDF document in Milvus with progress tracking and batch processing"""
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
            all_chunks = self.pdf_processor.process_pdf(pdf_path, document_id)
            
            if not all_chunks:
                logger.error(f"No chunks generated from PDF: {pdf_path}")
                return False
            
            total_chunks = len(all_chunks)
            logger.info(f"Generated {total_chunks} chunks, processing in batches of {self.batch_size}")
            
            # Process chunks in batches with progress bar
            successful_chunks = 0
            failed_chunks = 0
            
            with tqdm(total=total_chunks, desc="Processing chunks", unit="chunk") as pbar:
                for i in range(0, total_chunks, self.batch_size):
                    batch = all_chunks[i:i + self.batch_size]
                    
                    # Generate embeddings for this batch
                    chunks_with_embeddings = self.generate_embeddings_in_batches(batch)
                    
                    if not chunks_with_embeddings:
                        logger.warning(f"Failed to generate embeddings for batch {i//self.batch_size + 1}")
                        failed_chunks += len(batch)
                        pbar.update(len(batch))
                        continue
                    
                    # Validate embeddings
                    valid_chunks = []
                    for chunk in chunks_with_embeddings:
                        if self.embedding_generator.validate_embedding(chunk.get("embedding", [])):
                            valid_chunks.append(chunk)
                        else:
                            logger.warning(f"Invalid embedding for chunk {chunk.get('chunk_id')}")
                    
                    if not valid_chunks:
                        logger.warning(f"No valid chunks in batch {i//self.batch_size + 1}")
                        failed_chunks += len(batch)
                        pbar.update(len(batch))
                        continue
                    
                    # Store batch in Milvus
                    if self.milvus_client.insert_documents(valid_chunks):
                        successful_chunks += len(valid_chunks)
                        logger.debug(f"Successfully stored batch {i//self.batch_size + 1} with {len(valid_chunks)} chunks")
                    else:
                        logger.error(f"Failed to store batch {i//self.batch_size + 1}")
                        failed_chunks += len(valid_chunks)
                    
                    pbar.update(len(batch))
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.1)
            
            logger.info(f"PDF processing completed:")
            logger.info(f"  ✓ Successfully stored: {successful_chunks} chunks")
            logger.info(f"  ✗ Failed: {failed_chunks} chunks")
            logger.info(f"  📊 Success rate: {successful_chunks/(successful_chunks+failed_chunks)*100:.1f}%")
            
            # Validate text coverage
            # logger.info("Validating text coverage...")
            # validation_results = self.validate_text_coverage(pdf_path, all_chunks)
            
            # if validation_results:
            #     if validation_results.get('pages_missing') or validation_results.get('pages_with_missing_text'):
            #         logger.warning("⚠️  Text validation found issues - check text_validation_report.txt for details")
            #     else:
            #         logger.info("✅ Text validation passed - all text appears to be embedded")
            
            return successful_chunks > 0
            
        except Exception as e:
            logger.error(f"Failed to store PDF {pdf_path}: {e}")
            return False
    
    def store_multiple_pdfs_with_progress(self, pdf_directory: str) -> Dict[str, Dict[str, Any]]:
        """Store multiple PDFs from a directory with detailed progress tracking"""
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
            
            for pdf_file in tqdm(pdf_files, desc="Processing PDFs", unit="pdf"):
                pdf_path = os.path.join(pdf_directory, pdf_file)
                document_id = os.path.splitext(pdf_file)[0]
                
                start_time = time.time()
                success = self.store_pdf_with_progress(pdf_path, document_id)
                end_time = time.time()
                
                results[pdf_file] = {
                    "success": success,
                    "processing_time": end_time - start_time,
                    "document_id": document_id
                }
                
                if success:
                    logger.info(f"✓ Successfully stored: {pdf_file} (took {end_time - start_time:.1f}s)")
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
                },
                "batch_config": {
                    "batch_size": self.batch_size,
                    "embedding_batch_size": self.embedding_batch_size
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
    
    def validate_text_coverage(self, pdf_path: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate that all text from PDF is being embedded"""
        try:
            logger.info(f"Validating text coverage for PDF: {pdf_path}")
            validation_results = validate_pdf_processing(pdf_path, chunks)
            
            # Log validation summary
            logger.info("Text validation completed:")
            logger.info(f"  Original pages: {validation_results['total_pages_original']}")
            logger.info(f"  Processed pages: {validation_results['total_pages_processed']}")
            logger.info(f"  Total chunks: {validation_results['total_chunks']}")
            logger.info(f"  Missing pages: {len(validation_results['pages_missing'])}")
            logger.info(f"  Low coverage pages: {len(validation_results['pages_with_missing_text'])}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate text coverage: {e}")
            return {}
    
    def close(self):
        """Close the system"""
        try:
            self.milvus_client.close()
            logger.info("PDF to Milvus batch system closed")
        except Exception as e:
            logger.error(f"Failed to close system: {e}")

def main():
    """Main function for command line usage with batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Store PDF documents in Milvus with batch processing")
    parser.add_argument("--pdf", help="Path to a single PDF file")
    parser.add_argument("--directory", help="Path to directory containing PDF files")
    parser.add_argument("--clear", action="store_true", help="Clear all documents from Milvus")
    parser.add_argument("--stats", action="store_true", help="Show storage statistics")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of chunks to process at once (default: 100)")
    parser.add_argument("--embedding-batch-size", type=int, default=50, help="Number of texts to embed at once (default: 50)")
    
    args = parser.parse_args()
    
    # Initialize system with batch processing
    pdf_storage = PDFToMilvusBatch(
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size
    )
    
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
            logger.info(f"  Batch Config: {stats.get('batch_config', 'N/A')}")
        
        elif args.pdf:
            success = pdf_storage.store_pdf_with_progress(args.pdf)
            if success:
                logger.info("✓ PDF stored successfully!")
            else:
                logger.error("✗ Failed to store PDF")
        
        elif args.directory:
            results = pdf_storage.store_multiple_pdfs_with_progress(args.directory)
            successful = sum(1 for result in results.values() if result["success"])
            total = len(results)
            total_time = sum(result["processing_time"] for result in results.values())
            logger.info(f"✓ Successfully stored {successful}/{total} PDFs in {total_time:.1f}s")
        
        else:
            logger.info("No action specified. Use --help for usage information.")
    
    finally:
        pdf_storage.close()

if __name__ == "__main__":
    main() 