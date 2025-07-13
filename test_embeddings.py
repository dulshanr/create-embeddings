#!/usr/bin/env python3
"""
Test script to compare sentence-transformers and OpenAI embeddings.
"""

import os
import time
from src.embedding_generator import EmbeddingGenerator, OpenAIEmbeddingGenerator
from config import *

def test_embedding_generators():
    """Test both embedding generators"""
    
    test_texts = [
        "This is a test document about machine learning.",
        "Artificial intelligence is transforming the world.",
        "Natural language processing helps computers understand text.",
        "Deep learning models can process large amounts of data.",
        "Vector databases store embeddings for similarity search."
    ]
    
    print("Testing embedding generators...")
    print("=" * 50)
    
    # Test sentence-transformers
    print("\n1. Testing sentence-transformers embedding generator:")
    try:
        st_generator = EmbeddingGenerator()
        st_start = time.time()
        st_embeddings = st_generator.generate_embeddings(test_texts)
        st_time = time.time() - st_start
        
        print(f"   ✓ Generated {len(st_embeddings)} embeddings")
        print(f"   ✓ Dimension: {len(st_embeddings[0]) if st_embeddings else 0}")
        print(f"   ✓ Time taken: {st_time:.2f} seconds")
        print(f"   ✓ Model: {EMBEDDING_MODEL}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test OpenAI
    print("\n2. Testing OpenAI embedding generator:")
    try:
        if not os.getenv('OPENAI_API_KEY'):
            print("   ⚠️  OPENAI_API_KEY not set. Skipping OpenAI test.")
            print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        else:
            openai_generator = OpenAIEmbeddingGenerator()
            openai_start = time.time()
            openai_embeddings = openai_generator.generate_embeddings(test_texts)
            openai_time = time.time() - openai_start
            
            print(f"   ✓ Generated {len(openai_embeddings)} embeddings")
            print(f"   ✓ Dimension: {len(openai_embeddings[0]) if openai_embeddings else 0}")
            print(f"   ✓ Time taken: {openai_time:.2f} seconds")
            print(f"   ✓ Model: {OPENAI_MODEL}")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Compare dimensions
    print("\n3. Dimension comparison:")
    if st_embeddings and openai_embeddings:
        st_dim = len(st_embeddings[0])
        openai_dim = len(openai_embeddings[0])
        print(f"   Sentence-transformers: {st_dim} dimensions")
        print(f"   OpenAI: {openai_dim} dimensions")
        print(f"   Difference: {openai_dim - st_dim} dimensions")
        
        if st_dim != openai_dim:
            print("   ⚠️  Different dimensions! You'll need to recreate your Milvus collection.")
    
    print("\n" + "=" * 50)
    print("Test completed!")

def test_single_embedding():
    """Test single embedding generation"""
    
    test_text = "This is a single test sentence for embedding generation."
    
    print("\nTesting single embedding generation:")
    print("=" * 50)
    
    # Test sentence-transformers
    try:
        st_generator = EmbeddingGenerator()
        st_embedding = st_generator.generate_single_embedding(test_text)
        print(f"✓ Sentence-transformers embedding length: {len(st_embedding)}")
    except Exception as e:
        print(f"✗ Sentence-transformers error: {e}")
    
    # Test OpenAI
    try:
        if os.getenv('OPENAI_API_KEY'):
            openai_generator = OpenAIEmbeddingGenerator()
            openai_embedding = openai_generator.generate_single_embedding(test_text)
            print(f"✓ OpenAI embedding length: {len(openai_embedding)}")
        else:
            print("⚠️  OPENAI_API_KEY not set. Skipping OpenAI test.")
    except Exception as e:
        print(f"✗ OpenAI error: {e}")

if __name__ == "__main__":
    test_embedding_generators()
    test_single_embedding() 