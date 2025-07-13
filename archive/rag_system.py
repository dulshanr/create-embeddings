import os
import uuid
from typing import List, Dict, Any, Optional
import logging
from milvus_client import MilvusClient
from pdf_processor import PDFProcessor
from embedding_generator import EmbeddingGenerator
from config import *

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with all components"""
        self.milvus_client = MilvusClient()
        self.pdf_processor = PDFProcessor()
        self.embedding_generator = EmbeddingGenerator()
        
    def initialize(self) -> bool:
        """Initialize the RAG system by connecting to Milvus and creating collection"""
        try:
            # Connect to Milvus
            if not self.milvus_client.connect():
                logger.error("Failed to connect to Milvus")
                return False
            
            # Create collection
            if not self.milvus_client.create_collection():
                logger.error("Failed to create collection")
                return False
            
            logger.info("RAG system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def store_pdf(self, pdf_path: str, document_id: Optional[str] = None) -> bool:
        """Store a PDF document in the RAG system"""
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
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents based on a query"""
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search in Milvus
            results = self.milvus_client.search_similar(query_embedding, top_k)
            
            logger.info(f"Found {len(results)} relevant documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
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
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a specific document from the system"""
        try:
            # Note: This is a simplified implementation
            # In a production system, you'd want to implement proper deletion
            logger.warning("Document deletion not implemented in this version")
            return False
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def clear_all_documents(self) -> bool:
        """Clear all documents from the system"""
        try:
            return self.milvus_client.delete_collection()
        except Exception as e:
            logger.error(f"Failed to clear all documents: {e}")
            return False
    
    def close(self):
        """Close the RAG system"""
        try:
            self.milvus_client.close()
            logger.info("RAG system closed")
        except Exception as e:
            logger.error(f"Failed to close RAG system: {e}") 