#!/usr/bin/env python3
"""
Example usage of the RAG system for storing PDFs in Milvus and performing searches.
"""

import os
import sys
from rag_system import RAGSystem
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function demonstrating RAG system usage"""
    
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag_system = RAGSystem()
    
    if not rag_system.initialize():
        logger.error("Failed to initialize RAG system")
        return
    
    # Example 1: Store a PDF document
    pdf_path = "sample_document.pdf"  # Replace with your PDF path
    
    if os.path.exists(pdf_path):
        logger.info(f"Storing PDF: {pdf_path}")
        success = rag_system.store_pdf(pdf_path, document_id="sample_doc_001")
        
        if success:
            logger.info("PDF stored successfully!")
        else:
            logger.error("Failed to store PDF")
    else:
        logger.warning(f"PDF file not found: {pdf_path}")
        logger.info("Please place a PDF file named 'sample_document.pdf' in the current directory")
    
    # Example 2: Search for relevant documents
    query = "What is machine learning?"
    logger.info(f"Searching for: {query}")
    
    results = rag_system.search_documents(query, top_k=3)
    
    if results:
        logger.info(f"Found {len(results)} relevant documents:")
        for i, result in enumerate(results, 1):
            logger.info(f"\nResult {i}:")
            logger.info(f"  Document ID: {result['document_id']}")
            logger.info(f"  Page: {result['page_number']}")
            logger.info(f"  Chunk ID: {result['chunk_id']}")
            logger.info(f"  Similarity Score: {result['score']:.4f}")
            logger.info(f"  Text: {result['text'][:200]}...")
    else:
        logger.info("No relevant documents found")
    
    # Example 3: Get system statistics
    stats = rag_system.get_system_stats()
    logger.info("\nSystem Statistics:")
    logger.info(f"  Milvus Stats: {stats.get('milvus_stats', 'N/A')}")
    logger.info(f"  Embedding Dimension: {stats.get('embedding_dimension', 'N/A')}")
    logger.info(f"  Chunk Config: {stats.get('chunk_config', 'N/A')}")
    
    # Close the system
    rag_system.close()
    logger.info("RAG system closed")

def interactive_mode():
    """Interactive mode for testing the RAG system"""
    
    logger.info("Starting interactive RAG system...")
    rag_system = RAGSystem()
    
    if not rag_system.initialize():
        logger.error("Failed to initialize RAG system")
        return
    
    try:
        while True:
            print("\n" + "="*50)
            print("RAG System Interactive Mode")
            print("="*50)
            print("1. Store PDF document")
            print("2. Search documents")
            print("3. Show system stats")
            print("4. Clear all documents")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                pdf_path = input("Enter PDF file path: ").strip()
                if os.path.exists(pdf_path):
                    document_id = input("Enter document ID (or press Enter for auto-generated): ").strip()
                    if not document_id:
                        document_id = None
                    
                    success = rag_system.store_pdf(pdf_path, document_id)
                    if success:
                        print("✓ PDF stored successfully!")
                    else:
                        print("✗ Failed to store PDF")
                else:
                    print(f"✗ File not found: {pdf_path}")
            
            elif choice == "2":
                query = input("Enter search query: ").strip()
                top_k = input("Enter number of results (default 5): ").strip()
                try:
                    top_k = int(top_k) if top_k else 5
                except ValueError:
                    top_k = 5
                
                results = rag_system.search_documents(query, top_k)
                
                if results:
                    print(f"\nFound {len(results)} relevant documents:")
                    for i, result in enumerate(results, 1):
                        print(f"\n--- Result {i} ---")
                        print(f"Document ID: {result['document_id']}")
                        print(f"Page: {result['page_number']}")
                        print(f"Score: {result['score']:.4f}")
                        print(f"Text: {result['text'][:300]}...")
                else:
                    print("No relevant documents found")
            
            elif choice == "3":
                stats = rag_system.get_system_stats()
                print("\nSystem Statistics:")
                print(f"Milvus Stats: {stats.get('milvus_stats', 'N/A')}")
                print(f"Embedding Dimension: {stats.get('embedding_dimension', 'N/A')}")
                print(f"Chunk Config: {stats.get('chunk_config', 'N/A')}")
            
            elif choice == "4":
                confirm = input("Are you sure you want to clear all documents? (y/N): ").strip().lower()
                if confirm == 'y':
                    success = rag_system.clear_all_documents()
                    if success:
                        print("✓ All documents cleared!")
                    else:
                        print("✗ Failed to clear documents")
                else:
                    print("Operation cancelled")
            
            elif choice == "5":
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please enter 1-5.")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        rag_system.close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main() 