#!/usr/bin/env python3
"""
Simple script to validate that all text from a PDF is being embedded.
"""

import os
import sys
from pdf_to_milvus_batch import PDFToMilvusBatch
from pdf_processor import PDFProcessor

def validate_pdf_text_coverage(pdf_path: str):
    """Validate that all text from a PDF is being embedded"""
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print(f"🔍 Validating text coverage for: {pdf_path}")
    print("=" * 60)
    
    try:
        # Initialize the system
        pdf_storage = PDFToMilvusBatch()
        if not pdf_storage.initialize():
            print("❌ Failed to initialize system")
            return False
        
        # Process PDF into chunks (without storing in Milvus)
        pdf_processor = PDFProcessor()
        chunks = pdf_processor.process_pdf(pdf_path, "validation_doc")
        
        if not chunks:
            print("❌ No chunks generated from PDF")
            return False
        
        print(f"📄 Generated {len(chunks)} chunks from PDF")
        
        # Validate text coverage
        validation_results = pdf_storage.validate_text_coverage(pdf_path, chunks)
        
        if not validation_results:
            print("❌ Validation failed")
            return False
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Original pages: {validation_results['total_pages_original']}")
        print(f"Processed pages: {validation_results['total_pages_processed']}")
        print(f"Total chunks: {validation_results['total_chunks']}")
        print(f"Missing pages: {len(validation_results['pages_missing'])}")
        print(f"Low coverage pages: {len(validation_results['pages_with_missing_text'])}")
        
        # Check for issues
        if validation_results['pages_missing']:
            print(f"\n⚠️  WARNING: Missing pages: {validation_results['pages_missing']}")
            print("   This could indicate:")
            print("   - Pages with only images (no text)")
            print("   - Scanned pages without OCR")
            print("   - PDF text extraction issues")
        
        if validation_results['pages_with_missing_text']:
            print(f"\n⚠️  WARNING: Low coverage pages: {validation_results['pages_with_missing_text']}")
            print("   This could indicate:")
            print("   - Text being truncated during chunking")
            print("   - Chunk size too large")
            print("   - Text extraction issues")
        
        if not validation_results['pages_missing'] and not validation_results['pages_with_missing_text']:
            print("\n✅ EXCELLENT: All text appears to be properly embedded!")
            print("   - All pages were processed")
            print("   - Text coverage is good")
        
        # Show detailed results for first few pages
        print("\n" + "=" * 60)
        print("📋 DETAILED RESULTS (First 5 pages)")
        print("=" * 60)
        
        for detail in validation_results['validation_details'][:5]:
            status_icon = "✅" if detail['status'] == 'OK' else "⚠️" if detail['status'] == 'LOW_COVERAGE' else "❌"
            print(f"\n{status_icon} Page {detail['page']}:")
            print(f"   Status: {detail['status']}")
            print(f"   Original length: {detail['original_length']} characters")
            print(f"   Chunks found: {detail['chunks_found']}")
            print(f"   Text coverage: {detail['text_coverage']:.2%}")
            print(f"   Original preview: {detail['original_text_preview']}")
            print(f"   Chunk preview: {detail['combined_chunk_text_preview']}")
        
        if len(validation_results['validation_details']) > 5:
            print(f"\n... and {len(validation_results['validation_details']) - 5} more pages")
        
        # Check chunk statistics
        print("\n" + "=" * 60)
        print("📈 CHUNK STATISTICS")
        print("=" * 60)
        
        chunk_lengths = [len(chunk.get('text', '')) for chunk in chunks]
        if chunk_lengths:
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            min_length = min(chunk_lengths)
            max_length = max(chunk_lengths)
            
            print(f"Average chunk length: {avg_length:.0f} characters")
            print(f"Shortest chunk: {min_length} characters")
            print(f"Longest chunk: {max_length} characters")
            print(f"Total text in chunks: {sum(chunk_lengths)} characters")
        
        # Close system
        pdf_storage.close()
        
        print(f"\n📄 Validation report saved to: text_validation_report.txt")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python validate_pdf_embedding.py <pdf_path>")
        print("Example: python validate_pdf_embedding.py sample.pdf")
        return
    
    pdf_path = sys.argv[1]
    validate_pdf_text_coverage(pdf_path)

if __name__ == "__main__":
    main() 