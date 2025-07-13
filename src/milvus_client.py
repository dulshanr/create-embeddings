from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusClient:
    def __init__(self):
        self.collection_name = COLLECTION_NAME
        self.dimension = DIMENSION
        self.connection = None
        self.collection = None
        
    def connect(self):
        """Connect to Milvus server"""
        try:
            if MILVUS_USER and MILVUS_PASSWORD:
                connections.connect(
                    alias="default",
                    host=MILVUS_HOST,
                    port=MILVUS_PORT,
                    user=MILVUS_USER,
                    password=MILVUS_PASSWORD
                )
            else:
                connections.connect(
                    alias="default",
                    host=MILVUS_HOST,
                    port=MILVUS_PORT
                )
            logger.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False
    
    def create_collection(self):
        """Create collection with proper schema"""
        try:
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="chunk_id", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
            
            schema = CollectionSchema(fields=fields, description="PDF document chunks with embeddings")
            
            # Check if collection exists and has correct schema
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection {self.collection_name} already exists")
                self.collection = Collection(self.collection_name)
                
                # Verify schema matches
                existing_schema = self.collection.schema
                expected_field_names = ["id", "document_id", "page_number", "chunk_id", "text", "embedding"]
                existing_field_names = [field.name for field in existing_schema.fields]
                
                if existing_field_names != expected_field_names:
                    logger.warning(f"Collection schema mismatch. Expected: {expected_field_names}, Got: {existing_field_names}")
                    logger.info("Dropping existing collection and recreating with correct schema")
                    utility.drop_collection(self.collection_name)
                    self.collection = Collection(name=self.collection_name, schema=schema)
                    logger.info(f"Recreated collection: {self.collection_name}")
                else:
                    logger.info("Collection schema matches expected schema")
            else:
                self.collection = Collection(name=self.collection_name, schema=schema)
                logger.info(f"Created collection: {self.collection_name}")
            
            # Create index if it doesn't exist
            try:
                index_params = {
                    "metric_type": METRIC_TYPE,
                    "index_type": INDEX_TYPE,
                    "params": {"nlist": 1024}
                }
                
                self.collection.create_index(field_name="embedding", index_params=index_params)
                logger.info("Created index on embedding field")
            except Exception as e:
                logger.info(f"Index may already exist or failed to create: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def insert_documents(self, documents: List[Dict[str, Any]]):
        """Insert document chunks into collection"""
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return False
            
            if not documents:
                logger.warning("No documents to insert")
                return True
            
            # Validate and prepare data as list of dictionaries (correct format)
            valid_documents = []
            
            for i, doc in enumerate(documents):
                # Validate required fields
                if "document_id" not in doc:
                    logger.error(f"Missing document_id in document {i}")
                    continue
                if "page_number" not in doc:
                    logger.error(f"Missing page_number in document {i}")
                    continue
                if "chunk_id" not in doc:
                    logger.error(f"Missing chunk_id in document {i}")
                    continue
                if "text" not in doc:
                    logger.error(f"Missing text in document {i}")
                    continue
                if "embedding" not in doc:
                    logger.error(f"Missing embedding in document {i}")
                    continue
                
                # Ensure document_id is a string
                doc_id = str(doc["document_id"])
                if len(doc_id) > 100:  # Max length for VARCHAR(100)
                    doc_id = doc_id[:100]
                
                # Create document in correct format
                valid_doc = {
                    "document_id": doc_id,
                    "page_number": int(doc["page_number"]),
                    "chunk_id": int(doc["chunk_id"]),
                    "text": str(doc["text"])[:MAX_TEXT_LENGTH],  # Truncate if too long
                    "embedding": doc["embedding"]
                }
                
                # Append valid_doc to text file
                # try:
                #     with open("milvus_documents_log.txt", "a", encoding="utf-8") as f:
                #         f.write(f"Document: {valid_doc}\n")
                #         f.write(f"Document ID: {valid_doc['document_id']}\n")
                #         f.write(f"Page: {valid_doc['page_number']}\n")
                #         f.write(f"Chunk: {valid_doc['chunk_id']}\n")
                #         f.write(f"Text: {str(doc["text"])}...\n")  # First 200 chars
                #         f.write(f"Embedding length: {len(valid_doc['embedding'])}\n")
                #         f.write("-" * 50 + "\n")
                # except Exception as e:
                #     logger.warning(f"Failed to write to log file: {e}")
                
                valid_documents.append(valid_doc)
            
            if not valid_documents:
                logger.error("No valid documents to insert")
                return False
            
            # Debug: Log data types
            if valid_documents:
                first_doc = valid_documents[0]
                logger.debug(f"Data types - document_id: {type(first_doc['document_id'])}, page_number: {type(first_doc['page_number'])}, chunk_id: {type(first_doc['chunk_id'])}, text: {type(first_doc['text'])}, embedding: {type(first_doc['embedding'])}")
            
            # Insert data as list of dictionaries (correct format)
            self.collection.insert(valid_documents)
            self.collection.flush()
            logger.info(f"Inserted {len(valid_documents)} document chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            return False
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return []
            
            self.collection.load()
            
            search_params = {
                "metric_type": METRIC_TYPE,
                "params": {"nprobe": NPROBE}
            }
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["document_id", "page_number", "chunk_id", "text"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "score": hit.score,
                        "document_id": hit.entity.get("document_id"),
                        "page_number": hit.entity.get("page_number"),
                        "chunk_id": hit.entity.get("chunk_id"),
                        "text": hit.entity.get("text")
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            if not self.collection:
                return {"error": "Collection not initialized"}
            
            # Use num_entities instead of get_statistics
            row_count = self.collection.num_entities
            return {
                "row_count": row_count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def delete_collection(self):
        """Delete the collection"""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def close(self):
        """Close connection"""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect: {e}") 