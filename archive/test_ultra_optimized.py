#!/usr/bin/env python3
"""
Test script for Ultra Memory Optimized PDF Processor
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pdf_processor_memory_optimized import UltraMemoryOptimizedPDFProcessor
from ultra_optimized_config import *

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_memory_usage():
    """Test memory usage monitoring"""
    processor = UltraMemoryOptimizedPDFProcessor()
    
    logger.info("Testing memory usage monitoring...")
    
    # Get initial memory
    initial_memory = processor._get_memory_usage_mb()
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Test memory cleanup
    processor._force_memory_cleanup()
    
    # Get memory after cleanup
    after_cleanup_memory = processor._get_memory_usage_mb()
    logger.info(f"Memory after cleanup: {after_cleanup_memory:.2f} MB")
    
    # Get detailed memory info
    memory_info = processor.get_memory_usage_info()
    if memory_info:
        logger.info(f"Detailed memory info: {memory_info}")
    
    return True

def test_pdf_processing_simulation():
    """Test PDF processing simulation without actual PDF"""
    processor = UltraMemoryOptimizedPDFProcessor()
    
    logger.info("Testing PDF processing simulation...")
    
    # Simulate processing a large text
    large_text = "This is a test. " * 1000  # Create a large text
    
    logger.info(f"Processing text with {len(large_text)} characters...")
    
    # Test chunking
    chunks = list(processor.chunk_text_streaming(large_text, 0))
    logger.info(f"Created {len(chunks)} chunks")
    
    # Test memory usage during processing
    for i, chunk in enumerate(chunks[:10]):  # Test first 10 chunks
        logger.info(f"Chunk {i+1}: {len(chunk['text'])} characters")
        
        if i % 5 == 0:
            memory_mb = processor._get_memory_usage_mb()
            logger.info(f"Memory usage at chunk {i+1}: {memory_mb:.2f} MB")
            processor._force_memory_cleanup()
    
    return True

def test_configuration():
    """Test that configuration is loaded correctly"""
    logger.info("Testing configuration...")
    
    logger.info(f"Ultra Memory MB: {ULTRA_MEMORY_MB}")
    logger.info(f"Ultra Pages Per Batch: {ULTRA_PAGES_PER_BATCH}")
    logger.info(f"Ultra Batch Size: {ULTRA_BATCH_SIZE}")
    logger.info(f"Chunk Size: {CHUNK_SIZE}")
    logger.info(f"Chunk Overlap: {CHUNK_OVERLAP}")
    logger.info(f"Max Text Length: {MAX_TEXT_LENGTH}")
    
    return True

def main():
    """Run all tests"""
    logger.info("Starting Ultra Memory Optimized PDF Processor tests...")
    
    tests = [
        ("Configuration Test", test_configuration),
        ("Memory Usage Test", test_memory_usage),
        ("PDF Processing Simulation Test", test_pdf_processing_simulation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            start_time = time.time()
            success = test_func()
            end_time = time.time()
            
            if success:
                logger.info(f"✅ {test_name} PASSED ({end_time - start_time:.2f}s)")
                results[test_name] = True
            else:
                logger.error(f"❌ {test_name} FAILED")
                results[test_name] = False
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The ultra-optimized processor is ready to use.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the logs.")
        return False

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1) 