from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import logging
from config import *
import os

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize the embedding generator with a sentence transformer model"""
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            
            # Convert to list of lists
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding[0] if embedding else []
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            return []
    
    def add_embeddings_to_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add embeddings to document chunks"""
        try:
            if not chunks:
                return []
            
            # Extract texts for embedding
            texts = [chunk["text"] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            if not embeddings:
                logger.error("Failed to generate embeddings")
                return []
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk["embedding"] = embeddings[i]
                else:
                    logger.warning(f"Missing embedding for chunk {i}")
                    chunk["embedding"] = []
            
            logger.info(f"Added embeddings to {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to chunks: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        try:
            # Generate a test embedding to get dimension
            test_embedding = self.generate_single_embedding("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}")
            return 0
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate if an embedding is properly formatted"""
        try:
            if not embedding:
                return False
            
            # Check if it's a list of floats
            if not all(isinstance(x, (int, float)) for x in embedding):
                return False
            
            # Check dimension
            expected_dim = self.get_embedding_dimension()
            if expected_dim > 0 and len(embedding) != expected_dim:
                logger.warning(f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to validate embedding: {e}")
            return False 

class OpenAIEmbeddingGenerator:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """Initialize the OpenAI embedding generator"""
        try:
            import openai
            self.model_name = model_name
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            logger.info(f"Initialized OpenAI embedding generator with model: {model_name}")
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embedding generator: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using OpenAI API"""
        try:
            import openai
            response = openai.embeddings.create(
                input=texts,
                model=self.model_name
            )
            embeddings = [d.embedding for d in response.data]
            logger.info(f"Generated embeddings for {len(texts)} texts using OpenAI")
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return []
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI API"""
        try:
            import openai
            response = openai.embeddings.create(
                input=[text],
                model=self.model_name
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embedding for text: {e}")
            return []
    
    def add_embeddings_to_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add embeddings to document chunks using OpenAI"""
        try:
            if not chunks:
                return []
            
            # Extract texts for embedding
            texts = [chunk["text"] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            if not embeddings:
                logger.error("Failed to generate OpenAI embeddings")
                return []
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk["embedding"] = embeddings[i]
                else:
                    logger.warning(f"Missing embedding for chunk {i}")
                    chunk["embedding"] = []
            
            logger.info(f"Added OpenAI embeddings to {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to add OpenAI embeddings to chunks: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the OpenAI embeddings"""
        try:
            # Generate a test embedding to get dimension
            test_embedding = self.generate_single_embedding("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Failed to get OpenAI embedding dimension: {e}")
            return 0
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate if an OpenAI embedding is properly formatted"""
        try:
            if not embedding:
                return False
            
            # Check if it's a list of floats
            if not all(isinstance(x, (int, float)) for x in embedding):
                return False
            
            # Check dimension
            expected_dim = self.get_embedding_dimension()
            if expected_dim > 0 and len(embedding) != expected_dim:
                logger.warning(f"OpenAI embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to validate OpenAI embedding: {e}")
            return False 