import fitz  # PyMuPDF
import re
from typing import List, Dict, Any
import logging
from config import *

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF and return page-wise content"""
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Clean text
                text = self._clean_text(text)
                
                if text.strip():  # Only add non-empty pages
                    pages.append({
                        "page_number": page_num + 1,
                        "text": text,
                        "page_width": page.rect.width,
                        "page_height": page.rect.height
                    })
            
            doc.close()
            logger.info(f"Extracted text from {len(pages)} pages")
            return pages
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_id: int = 0) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        if len(words) <= self.chunk_size:
            # If text is shorter than chunk size, return as single chunk
            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "start_word": 0,
                "end_word": len(words)
            })
        else:
            # Create overlapping chunks
            start = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                
                chunk_text = ' '.join(words[start:end])
                
                chunks.append({
                    "chunk_id": chunk_id + len(chunks),
                    "text": chunk_text,
                    "start_word": start,
                    "end_word": end
                })
                
                # Move start position with overlap
                start = end - self.chunk_overlap
                if start >= len(words):
                    break
        
        return chunks
    
    def process_pdf(self, pdf_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Process PDF and return chunks ready for embedding"""
        try:
            # Extract text from PDF
            pages = self.extract_text_from_pdf(pdf_path)
            
            if not pages:
                logger.warning(f"No text extracted from {pdf_path}")
                return []
            
            # Process each page
            all_chunks = []
            chunk_counter = 0
            
            for page in pages:
                # Chunk the page text
                page_chunks = self.chunk_text(page["text"], chunk_counter)
                
                # Add metadata to each chunk
                for chunk in page_chunks:
                    chunk.update({
                        "document_id": document_id,
                        "page_number": page["page_number"],
                        "page_width": page["page_width"],
                        "page_height": page["page_height"]
                    })
                    all_chunks.append(chunk)
                
                chunk_counter += len(page_chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            return []
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        text_lengths = [len(chunk["text"]) for chunk in chunks]
        word_counts = [len(chunk["text"].split()) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(text_lengths) / len(text_lengths),
            "min_chunk_length": min(text_lengths),
            "max_chunk_length": max(text_lengths),
            "avg_word_count": sum(word_counts) / len(word_counts),
            "min_word_count": min(word_counts),
            "max_word_count": max(word_counts)
        } 