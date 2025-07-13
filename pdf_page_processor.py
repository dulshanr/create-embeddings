#!/usr/bin/env python3
"""
Simple PDF Page-by-Page Processor
Processes large PDFs page by page to minimize memory usage
"""

import fitz  # PyMuPDF
import gc
import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.milvus_client import MilvusClient
from src.embedding_generator import EmbeddingGenerator
from config import *

# Import LangChain text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/page_processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PDFPageProcessor:
    def __init__(self):
        self.milvus_client = MilvusClient()
        
        # Choose embedding generator based on configuration
        if USE_OPENAI_EMBEDDINGS:
            from src.embedding_generator import OpenAIEmbeddingGenerator
            self.embedding_generator = OpenAIEmbeddingGenerator(OPENAI_MODEL)
            logger.info(f"Using OpenAI embeddings with model: {OPENAI_MODEL}")
        else:
            self.embedding_generator = EmbeddingGenerator()
            logger.info(f"Using sentence-transformers with model: {EMBEDDING_MODEL}")
        
        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize Milvus connection and collection
        if not self.milvus_client.connect():
            logger.error("Failed to connect to Milvus")
            raise Exception("Milvus connection failed")
        
        if not self.milvus_client.create_collection():
            logger.error("Failed to create Milvus collection")
            raise Exception("Milvus collection creation failed")
        
        logger.info("PDF Page Processor initialized successfully")
        
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def process_pdf_page_by_page(self, pdf_path: str, document_id: str = None) -> bool:
        """Process PDF page by page with immediate memory cleanup"""
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            if document_id is None:
                document_id = Path(pdf_path).stem
            
            # Get file size
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            logger.info(f"Processing PDF: {pdf_path}")
            logger.info(f"File size: {file_size_mb:.2f} MB")
            logger.info(f"Document ID: {document_id}")
            
            # Get initial memory
            initial_memory = self.get_memory_usage_mb()
            logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
            
            # Open PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            logger.info(f"PDF has {total_pages} pages")
            
            start_time = time.time()
            total_embeddings = 0
            
            # Process each page individually
            for page_num in range(total_pages):
                try:
                    # Get memory before processing page
                    memory_before = self.get_memory_usage_mb()
                    
                    # Load single page
                    page = doc.load_page(page_num)
                    
                    # Extract text from page
                    text = page.get_text()
                    
                    # Clean text
                    text = self._clean_text(text)
                    
                    if text.strip():  # Only process non-empty pages
                        # Chunk the text from this page using LangChain
                        page_chunks = self._chunk_text_langchain(text, page_num)
                        
                        for chunk_idx, chunk_text in enumerate(page_chunks):
                            # Generate embedding for this chunk
                            embedding = self.embedding_generator.generate_single_embedding(chunk_text)
                            
                            if embedding:
                                # Prepare data for Milvus
                                chunk_id = page_num * 1000 + chunk_idx  # Unique chunk ID
                                data = {
                                    "document_id": document_id,
                                    "page_number": page_num + 1,
                                    "chunk_id": chunk_id,
                                    "text": chunk_text,
                                    "embedding": embedding,
                                    "page_width": page.rect.width,
                                    "page_height": page.rect.height
                                }
                                
                                # Save data to log file
                                try:
                                    with open("logs/page_data_log.txt", "a", encoding="utf-8") as f:
                                        f.write(f"\n{'='*60}\n")
                                        f.write(f"Page {page_num + 1}/{total_pages} - Chunk {chunk_idx + 1}/{len(page_chunks)} - {document_id}\n")
                                        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                                        f.write(f"Document ID: {data['document_id']}\n")
                                        f.write(f"Page Number: {data['page_number']}\n")
                                        f.write(f"Chunk ID: {data['chunk_id']}\n")
                                        f.write(f"Page Width: {data['page_width']}\n")
                                        f.write(f"Page Height: {data['page_height']}\n")
                                        f.write(f"Text Length: {len(data['text'])} characters\n")
                                        f.write(f"Text Preview: {data['text'][:200]}...\n")
                                        f.write(f"Embedding Length: {len(data['embedding'])}\n")
                                        f.write(f"Embedding Preview: {data['embedding'][:5]}...\n")
                                        f.write(f"{'='*60}\n")
                                except Exception as e:
                                    logger.warning(f"Failed to write data to log file: {e}")
                                
                                # Insert into Milvus
                                success = self.milvus_client.insert_documents([data])
                                if success:
                                    total_embeddings += 1
                                    logger.info(f"✅ Page {page_num + 1}/{total_pages} Chunk {chunk_idx + 1}/{len(page_chunks)}: Inserted embedding")
                                else:
                                    logger.error(f"❌ Page {page_num + 1}/{total_pages} Chunk {chunk_idx + 1}/{len(page_chunks)}: Failed to insert")
                            else:
                                logger.warning(f"⚠️ Page {page_num + 1}/{total_pages} Chunk {chunk_idx + 1}/{len(page_chunks)}: No embedding generated")
                    else:
                        logger.info(f"📄 Page {page_num + 1}/{total_pages}: Empty page, skipped")
                    
                    # Immediately delete page object
                    del page
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Get memory after processing page
                    memory_after = self.get_memory_usage_mb()
                    
                    # Log progress every 10 pages
                    if (page_num + 1) % 10 == 0:
                        elapsed_time = time.time() - start_time
                        pages_per_second = (page_num + 1) / elapsed_time
                        logger.info(f"📊 Progress: {page_num + 1}/{total_pages} pages "
                                  f"({pages_per_second:.1f} pages/sec) "
                                  f"Memory: {memory_after:.2f} MB")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing page {page_num + 1}: {e}")
                    continue
            
            # Close document
            doc.close()
            
            # Final cleanup
            gc.collect()
            
            # Summary
            total_time = time.time() - start_time
            final_memory = self.get_memory_usage_mb()
            
            logger.info(f"\n{'='*50}")
            logger.info("PROCESSING COMPLETED!")
            logger.info(f"{'='*50}")
            logger.info(f"Total pages processed: {total_pages}")
            logger.info(f"Total embeddings inserted: {total_embeddings}")
            logger.info(f"Total processing time: {total_time:.1f} seconds")
            logger.info(f"Average speed: {total_pages/total_time:.1f} pages/second")
            logger.info(f"Final memory usage: {final_memory:.2f} MB")
            logger.info(f"Memory change: {final_memory - initial_memory:+.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            return False
    
    def _chunk_text_langchain(self, text: str, page_num: int) -> List[str]:
        """Split text into chunks using LangChain's RecursiveCharacterTextSplitter"""
        try:
            # Use LangChain's text splitter for memory-efficient chunking
            chunks = self.text_splitter.split_text(text)
            
            logger.info(f"Page {page_num + 1}: Split into {len(chunks)} chunks using LangChain")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking text for page {page_num + 1}: {e}")
            # Fallback to single chunk if splitting fails
            return [text]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
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
                logger.info(f"\n{'='*50}")
                logger.info(f"Processing PDF {i}/{len(pdf_files)}: {pdf_file.name}")
                logger.info(f"{'='*50}")
                
                success = self.process_pdf_page_by_page(str(pdf_file))
                results[pdf_file.name] = success
                
                if success:
                    logger.info(f"✅ Successfully processed: {pdf_file.name}")
                else:
                    logger.error(f"❌ Failed to process: {pdf_file.name}")
                
                # Wait between files
                if i < len(pdf_files):
                    logger.info("Waiting 2 seconds before next file...")
                    time.sleep(2)
            
            # Summary
            successful = sum(1 for success in results.values() if success)
            logger.info(f"\n{'='*50}")
            logger.info("FINAL SUMMARY")
            logger.info(f"{'='*50}")
            logger.info(f"Successfully processed: {successful}/{len(pdf_files)} PDFs")
            
        except Exception as e:
            logger.error(f"Error processing multiple PDFs: {e}")
        
        return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Page-by-Page Processor")
    parser.add_argument("--pdf", type=str, help="Path to single PDF file")
    parser.add_argument("--directory", type=str, help="Path to directory containing PDFs")
    parser.add_argument("--document-id", type=str, help="Custom document ID (for single PDF)")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    processor = PDFPageProcessor()
    
    if args.pdf:
        # Process single PDF
        success = processor.process_pdf_page_by_page(args.pdf, args.document_id)
        if success:
            logger.info("🎉 PDF processing completed successfully!")
        else:
            logger.error("❌ PDF processing failed!")
            sys.exit(1)
    
    elif args.directory:
        # Process multiple PDFs
        results = processor.process_multiple_pdfs(args.directory)
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        if successful == total:
            logger.info("🎉 All PDFs processed successfully!")
        else:
            logger.warning(f"⚠️ Only {successful}/{total} PDFs processed successfully")
            sys.exit(1)
    
    else:
        # Default: process PDFs in the 'in' directory
        if os.path.exists("in"):
            results = processor.process_multiple_pdfs("in")
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            
            if successful == total:
                logger.info("🎉 All PDFs processed successfully!")
            else:
                logger.warning(f"⚠️ Only {successful}/{total} PDFs processed successfully")
                sys.exit(1)
        else:
            logger.error("No PDF file or directory specified, and 'in' directory not found")
            parser.print_help()
            sys.exit(1)

if __name__ == "__main__":
    main() 