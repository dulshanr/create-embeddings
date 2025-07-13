#!/usr/bin/env python3
"""
Script to test if text is being truncated during chunk processing and embedding.
"""

from pdf_processor import PDFProcessor
from embedding_generator import EmbeddingGenerator
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from config import *
import os

def test_chunk_truncation():
    """Test if chunks are being truncated during processing"""
    
    print("🔍 Testing chunk truncation during processing")
    print("=" * 60)
    
    try:
        # Initialize components
        pdf_processor = PDFProcessor()
        embedding_generator = EmbeddingGenerator()
        
        # Get tokenizer info
        model = SentenceTransformer(EMBEDDING_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        
        print(f"📊 Model info:")
        print(f"  - Model: {EMBEDDING_MODEL}")
        print(f"  - Max tokens: {tokenizer.model_max_length}")
        print(f"  - Chunk size: {CHUNK_SIZE} characters")
        print(f"  - Chunk overlap: {CHUNK_OVERLAP} characters")
        
        # Create test text
        test_text = """
        This is a test document with multiple paragraphs to simulate real PDF content.
        The goal is to see if text gets truncated during the chunking and embedding process.
        
        This second paragraph continues the content and adds more text to make the document longer.
        We want to see how the tokenizer handles this text and whether any content is lost.
        
        A third paragraph adds even more content to test the limits further.
        This helps us understand the practical implications of token limits on real-world document processing.
        The tokenizer will process this text and we can see how it handles the length and complexity.
        
        This fourth paragraph continues to add content to make the text longer and more realistic.
        We're testing to see if the chunking process preserves all the text or if some gets truncated.
        The embedding process should handle this text properly without losing important information.
        
        Finally, this fifth paragraph wraps up the test content with additional text to ensure we have enough
        content to test the chunking and embedding processes thoroughly.
        """ * 5  # Repeat to make it longer
        
        print(f"\n📄 Test text length: {len(test_text)} characters")
        
        # Test tokenization
        tokens = tokenizer.encode(test_text, truncation=True, max_length=tokenizer.model_max_length)
        full_tokens = tokenizer.encode(test_text, truncation=False)
        
        print(f"Token analysis:")
        print(f"  - Full text tokens: {len(full_tokens)}")
        print(f"  - Truncated tokens: {len(tokens)}")
        print(f"  - Truncation ratio: {len(tokens)/len(full_tokens)*100:.1f}%")
        
        if len(full_tokens) > tokenizer.model_max_length:
            print(f"⚠️  WARNING: Text would be truncated by tokenizer")
        else:
            print(f"✅ Text fits within tokenizer limits")
        
        # Test chunking
        print(f"\n🧩 Testing chunking process:")
        print("-" * 40)
        
        # Create a temporary PDF-like structure
        chunks = []
        for i in range(0, len(test_text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = test_text[i:i + CHUNK_SIZE]
            if chunk_text.strip():
                chunk = {
                    "text": chunk_text,
                    "chunk_id": f"test_chunk_{len(chunks)}",
                    "page_number": 1,
                    "document_id": "test_doc"
                }
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} chunks")
        
        # Test each chunk for truncation
        total_original_chars = 0
        total_tokenized_chars = 0
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            total_original_chars += len(chunk_text)
            
            # Tokenize the chunk
            chunk_tokens = tokenizer.encode(chunk_text, truncation=True, max_length=tokenizer.model_max_length)
            full_chunk_tokens = tokenizer.encode(chunk_text, truncation=False)
            
            was_truncated = len(full_chunk_tokens) > tokenizer.model_max_length
            truncation_ratio = len(chunk_tokens) / len(full_chunk_tokens) if full_chunk_tokens else 1.0
            
            total_tokenized_chars += len(chunk_text) * truncation_ratio
            
            status = "⚠️" if was_truncated else "✅"
            print(f"{status} Chunk {i+1}: {len(chunk_text)} chars → {len(chunk_tokens)} tokens")
            
            if was_truncated:
                print(f"   Truncation: {truncation_ratio*100:.1f}% of text preserved")
        
        # Test embedding generation
        print(f"\n🔍 Testing embedding generation:")
        print("-" * 40)
        
        texts = [chunk["text"] for chunk in chunks]
        try:
            embeddings = embedding_generator.generate_embeddings(texts)
            print(f"✅ Successfully generated embeddings for {len(embeddings)} chunks")
            
            # Check if any embeddings failed
            failed_embeddings = [i for i, emb in enumerate(embeddings) if not emb]
            if failed_embeddings:
                print(f"⚠️  Failed embeddings for chunks: {failed_embeddings}")
            else:
                print(f"✅ All embeddings generated successfully")
                
        except Exception as e:
            print(f"❌ Failed to generate embeddings: {e}")
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print("=" * 40)
        print(f"Total original characters: {total_original_chars}")
        print(f"Estimated preserved characters: {total_tokenized_chars:.0f}")
        print(f"Overall preservation: {total_tokenized_chars/total_original_chars*100:.1f}%")
        
        if total_tokenized_chars < total_original_chars * 0.95:
            print(f"⚠️  WARNING: Significant text loss detected")
            print(f"   Consider reducing CHUNK_SIZE in config.py")
        else:
            print(f"✅ Text preservation is good")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing chunk truncation: {e}")
        return False

def test_with_real_pdf_chunks(pdf_path: str):
    """Test with actual PDF chunks"""
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print(f"\n📄 Testing with real PDF: {pdf_path}")
    print("=" * 60)
    
    try:
        pdf_processor = PDFProcessor()
        embedding_generator = EmbeddingGenerator()
        
        # Process PDF
        chunks = pdf_processor.process_pdf(pdf_path, "test_doc")
        
        if not chunks:
            print("❌ No chunks generated from PDF")
            return False
        
        print(f"Generated {len(chunks)} chunks from PDF")
        
        # Analyze chunks
        total_chars = 0
        truncated_chunks = 0
        
        for i, chunk in enumerate(chunks[:10]):  # Check first 10 chunks
            chunk_text = chunk.get("text", "")
            total_chars += len(chunk_text)
            
            # Check if this chunk would be truncated
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
            tokens = tokenizer.encode(chunk_text, truncation=True, max_length=tokenizer.model_max_length)
            full_tokens = tokenizer.encode(chunk_text, truncation=False)
            
            if len(full_tokens) > tokenizer.model_max_length:
                truncated_chunks += 1
                print(f"⚠️  Chunk {i+1} would be truncated: {len(chunk_text)} chars")
                print(f"   Tokens: {len(full_tokens)} → {len(tokens)} ({len(tokens)/len(full_tokens)*100:.1f}%)")
        
        print(f"\n📊 Real PDF Analysis:")
        print(f"  - Total chunks: {len(chunks)}")
        print(f"  - Chunks checked: 10")
        print(f"  - Truncated chunks: {truncated_chunks}")
        print(f"  - Average chunk length: {total_chars/10:.0f} characters")
        
        if truncated_chunks > 0:
            print(f"⚠️  {truncated_chunks} chunks would be truncated")
            print(f"   Consider reducing CHUNK_SIZE in config.py")
        else:
            print(f"✅ All checked chunks are within token limits")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing with real PDF: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # Test with synthetic text
    test_chunk_truncation()
    
    # Test with real PDF if provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        test_with_real_pdf_chunks(pdf_path)
    else:
        print(f"\n💡 To test with a real PDF, run:")
        print(f"   python test_chunk_truncation.py your_pdf_file.pdf") 