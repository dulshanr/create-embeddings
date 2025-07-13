import fitz  # PyMuPDF
import re
import gc
import os
from typing import List, Dict, Any, Generator, Iterator
import logging
from config import *

logger = logging.getLogger(__name__)

class UltraMemoryOptimizedPDFProcessor:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.max_pages_per_batch = 5  # Process only 5 pages at a time
        self.max_memory_mb = 500  # Target max memory usage in MB
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _force_memory_cleanup(self):
        """Force aggressive memory cleanup"""
        gc.collect()
        gc.collect()  # Call twice for better cleanup
        
        # Log memory usage
        memory_mb = self._get_memory_usage_mb()
        if memory_mb > 0:
            logger.info(f"Memory usage after cleanup: {memory_mb:.2f} MB")
    
    def extract_text_from_pdf_streaming(self, pdf_path: str) -> Generator[Dict[str, Any], None, None]:
        """Extract text from PDF using streaming approach to minimize memory usage"""
        try:
            # Check file size first
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            logger.info(f"PDF file size: {file_size_mb:.2f} MB")
            
            if file_size_mb > 100:  # Large file warning
                logger.warning(f"Large PDF detected ({file_size_mb:.2f} MB). Using ultra-conservative memory approach.")
            
            # Open PDF with minimal memory footprint
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            logger.info(f"Processing PDF with {total_pages} pages")
            
            # Process pages in very small batches
            for batch_start in range(0, total_pages, self.max_pages_per_batch):
                batch_end = min(batch_start + self.max_pages_per_batch, total_pages)
                
                logger.info(f"Processing pages {batch_start + 1} to {batch_end}")
                
                for page_num in range(batch_start, batch_end):
                    try:
                        # Load only one page at a time
                        page = doc.load_page(page_num)
                        text = page.get_text()
                        
                        # Clean text immediately
                        text = self._clean_text(text)
                        
                        if text.strip():  # Only yield non-empty pages
                            page_data = {
                                "page_number": page_num + 1,
                                "text": text,
                                "page_width": page.rect.width,
                                "page_height": page.rect.height
                            }
                            
                            yield page_data
                        
                        # Immediately delete page object
                        del page
                        
                        # Check memory usage every few pages
                        if (page_num + 1) % 10 == 0:
                            memory_mb = self._get_memory_usage_mb()
                            if memory_mb > self.max_memory_mb:
                                logger.warning(f"High memory usage detected: {memory_mb:.2f} MB")
                                self._force_memory_cleanup()
                        
                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                # Force cleanup after each batch
                self._force_memory_cleanup()
                
                # Log progress
                logger.info(f"Completed batch: pages {batch_start + 1} to {batch_end}")
            
            doc.close()
            logger.info(f"Completed processing {total_pages} pages")
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
            return
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text with minimal memory usage"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def chunk_text_streaming(self, text: str, chunk_id: int = 0) -> Iterator[Dict[str, Any]]:
        """Split text into overlapping chunks using streaming approach"""
        if not text.strip():
            return
        
        words = text.split()
        
        if len(words) <= self.chunk_size:
            # If text is shorter than chunk size, return as single chunk
            yield {
                "chunk_id": chunk_id,
                "text": text,
                "start_word": 0,
                "end_word": len(words)
            }
        else:
            # Create overlapping chunks one at a time
            start = 0
            chunk_counter = 0
            
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                
                chunk_text = ' '.join(words[start:end])
                
                yield {
                    "chunk_id": chunk_id + chunk_counter,
                    "text": chunk_text,
                    "start_word": start,
                    "end_word": end
                }
                
                chunk_counter += 1
                
                # Move start position with overlap
                start = end - self.chunk_overlap
                if start >= len(words):
                    break
    
    def process_pdf_ultra_optimized(self, pdf_path: str, document_id: str) -> Generator[List[Dict[str, Any]], None, None]:
        """Process PDF with ultra-conservative memory management"""
        try:
            chunk_counter = 0
            page_count = 0
            batch_chunks = []
            batch_size = 20  # Small batch size for memory control
            
            # Process pages one at a time
            for page in self.extract_text_from_pdf_streaming(pdf_path):
                page_count += 1
                
                # Chunk the page text using streaming
                for chunk in self.chunk_text_streaming(page["text"], chunk_counter):
                    # Add metadata to chunk
                    chunk.update({
                        "document_id": document_id,
                        "page_number": page["page_number"],
                        "page_width": page["page_width"],
                        "page_height": page["page_height"]
                    })
                    
                    batch_chunks.append(chunk)
                    chunk_counter += 1
                    
                    # Yield when batch is full
                    if len(batch_chunks) >= batch_size:
                        yield batch_chunks
                        batch_chunks = []
                        
                        # Force memory cleanup
                        self._force_memory_cleanup()
                
                # Log progress every 10 pages
                if page_count % 10 == 0:
                    memory_mb = self._get_memory_usage_mb()
                    logger.info(f"Processed {page_count} pages, created {chunk_counter} chunks. Memory: {memory_mb:.2f} MB")
            
            # Yield remaining chunks
            if batch_chunks:
                yield batch_chunks
            
            logger.info(f"Completed processing: {page_count} pages, {chunk_counter} total chunks")
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            return
    
    def process_pdf_with_memory_monitoring(self, pdf_path: str, document_id: str, 
                                         max_memory_mb: int = None) -> Generator[List[Dict[str, Any]], None, None]:
        """Process PDF with continuous memory monitoring and automatic cleanup"""
        if max_memory_mb is None:
            max_memory_mb = self.max_memory_mb
        
        try:
            initial_memory = self._get_memory_usage_mb()
            logger.info(f"Starting PDF processing. Initial memory: {initial_memory:.2f} MB")
            
            for batch in self.process_pdf_ultra_optimized(pdf_path, document_id):
                current_memory = self._get_memory_usage_mb()
                
                # If memory usage is too high, force cleanup
                if current_memory > max_memory_mb:
                    logger.warning(f"Memory usage {current_memory:.2f} MB exceeds limit {max_memory_mb} MB. Forcing cleanup.")
                    self._force_memory_cleanup()
                
                yield batch
                
                # Small delay to allow system to stabilize
                import time
                time.sleep(0.01)
            
            final_memory = self._get_memory_usage_mb()
            logger.info(f"PDF processing completed. Final memory: {final_memory:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to process PDF with memory monitoring {pdf_path}: {e}")
            return
    
    def get_memory_usage_info(self) -> Dict[str, Any]:
        """Get detailed memory usage information"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024
            }
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return {}

# Keep the original class for backward compatibility
class MemoryOptimizedPDFProcessor(UltraMemoryOptimizedPDFProcessor):
    """Backward compatibility wrapper"""
    pass 