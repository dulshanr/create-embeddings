#!/usr/bin/env python3
"""
Script to check tokenizer limits and their impact on text embedding.
"""

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import os
from config import EMBEDDING_MODEL

def check_tokenizer_limits():
    """Check the tokenizer limits for the embedding model"""
    
    print(f"🔍 Checking tokenizer limits for model: {EMBEDDING_MODEL}")
    print("=" * 60)
    
    try:
        # Load the model
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Get the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model info:")
        print(f"   - Model name: {EMBEDDING_MODEL}")
        print(f"   - Embedding dimension: {model.get_sentence_embedding_dimension()}")
        
        # Check tokenizer limits
        max_length = tokenizer.model_max_length
        print(f"   - Max token length: {max_length}")
        print(f"   - Tokenizer name: {tokenizer.__class__.__name__}")
        
        # Test with different text lengths
        test_texts = [
            "Short text",
            "This is a medium length text that should fit comfortably within the token limit.",
            "A" * 1000,  # 1000 characters
            "A" * 2000,  # 2000 characters
            "A" * 4000,  # 4000 characters
            "A" * 8000,  # 8000 characters
            "A" * 16000, # 16000 characters
        ]
        
        print(f"\n🧪 Testing text length limits:")
        print("-" * 60)
        
        for i, text in enumerate(test_texts):
            # Tokenize the text
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            token_count = len(tokens)
            
            # Check if text was truncated
            full_tokens = tokenizer.encode(text, truncation=False)
            was_truncated = len(full_tokens) > max_length
            
            status = "✅" if not was_truncated else "⚠️"
            truncation_info = " (truncated)" if was_truncated else ""
            
            print(f"{status} Text {i+1} ({len(text)} chars): {token_count} tokens{truncation_info}")
            
            if was_truncated:
                print(f"   Original would be: {len(full_tokens)} tokens")
                print(f"   Truncation ratio: {token_count/len(full_tokens)*100:.1f}%")
        
        # Test with realistic text
        print(f"\n📄 Testing with realistic text:")
        print("-" * 60)
        
        realistic_text = """
        This is a sample document that contains multiple paragraphs of text. 
        It simulates the kind of content you might find in a PDF document. 
        The text includes various elements like sentences, paragraphs, and different types of content.
        
        This second paragraph continues the document with more content. 
        It demonstrates how the tokenizer handles longer pieces of text that might be extracted from PDF pages.
        The goal is to understand how much text can be processed before hitting token limits.
        
        A third paragraph adds even more content to test the limits further. 
        This helps us understand the practical implications of token limits on real-world document processing.
        The tokenizer will process this text and we can see how it handles the length and complexity.
        """ * 10  # Repeat to make it longer
        
        tokens = tokenizer.encode(realistic_text, truncation=True, max_length=max_length)
        full_tokens = tokenizer.encode(realistic_text, truncation=False)
        was_truncated = len(full_tokens) > max_length
        
        print(f"Realistic text ({len(realistic_text)} chars): {len(tokens)} tokens")
        if was_truncated:
            print(f"   Original would be: {len(full_tokens)} tokens")
            print(f"   Truncation ratio: {len(tokens)/len(full_tokens)*100:.1f}%")
        
        # Calculate practical limits
        print(f"\n📏 Practical limits for your use case:")
        print("-" * 60)
        
        # Estimate characters per token (rough approximation)
        avg_chars_per_token = len(realistic_text) / len(full_tokens)
        max_chars = max_length * avg_chars_per_token
        
        print(f"Estimated max characters per chunk: {max_chars:.0f}")
        print(f"Average characters per token: {avg_chars_per_token:.1f}")
        print(f"Your current chunk size: {CHUNK_SIZE} characters")
        
        if CHUNK_SIZE > max_chars:
            print(f"⚠️  WARNING: Your chunk size ({CHUNK_SIZE}) exceeds estimated limit ({max_chars:.0f})")
            print(f"   Consider reducing CHUNK_SIZE in config.py")
        else:
            print(f"✅ Your chunk size ({CHUNK_SIZE}) is within limits")
        
        # Test embedding generation
        print(f"\n🔍 Testing embedding generation:")
        print("-" * 60)
        
        test_texts_for_embedding = [
            "Short text for embedding",
            "A" * 1000,
            "A" * 4000,
            realistic_text[:max_chars] if len(realistic_text) > max_chars else realistic_text
        ]
        
        for i, text in enumerate(test_texts_for_embedding):
            try:
                embedding = model.encode([text])[0]
                print(f"✅ Text {i+1} ({len(text)} chars): Embedding generated ({len(embedding)} dimensions)")
            except Exception as e:
                print(f"❌ Text {i+1} ({len(text)} chars): Failed - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking tokenizer limits: {e}")
        return False

def check_current_chunk_processing():
    """Check how your current chunk processing might be affected"""
    
    print(f"\n" + "=" * 60)
    print("📋 CURRENT CONFIGURATION ANALYSIS")
    print("=" * 60)
    
    from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_TEXT_LENGTH
    
    print(f"Current settings:")
    print(f"  - CHUNK_SIZE: {CHUNK_SIZE} characters")
    print(f"  - CHUNK_OVERLAP: {CHUNK_OVERLAP} characters")
    print(f"  - MAX_TEXT_LENGTH: {MAX_TEXT_LENGTH} characters")
    
    # Estimate tokens
    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    
    # Test with chunk-sized text
    test_chunk = "A" * CHUNK_SIZE
    tokens = tokenizer.encode(test_chunk, truncation=True, max_length=tokenizer.model_max_length)
    
    print(f"\nToken analysis:")
    print(f"  - Your chunk size ({CHUNK_SIZE} chars) ≈ {len(tokens)} tokens")
    print(f"  - Tokenizer max length: {tokenizer.model_max_length} tokens")
    print(f"  - Available headroom: {tokenizer.model_max_length - len(tokens)} tokens")
    
    if len(tokens) > tokenizer.model_max_length * 0.8:
        print(f"⚠️  WARNING: Your chunks are using >80% of token limit")
        print(f"   Consider reducing CHUNK_SIZE to leave more headroom")
    else:
        print(f"✅ Your chunk size is well within token limits")

if __name__ == "__main__":
    check_tokenizer_limits()
    check_current_chunk_processing() 