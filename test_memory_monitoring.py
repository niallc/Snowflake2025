#!/usr/bin/env python3
"""
Test script for memory monitoring functionality.
"""

import psutil
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_memory_usage():
    """Check if memory usage is too high."""
    try:
        memory_percent = psutil.virtual_memory().percent
        logger.info(f"Current memory usage: {memory_percent:.1f}%")
        if memory_percent > 80:
            logger.error(f"Memory usage too high: {memory_percent:.1f}% > 80%")
            return True
        return False
    except Exception as e:
        logger.warning(f"Could not check memory usage: {e}")
        return False

def test_memory_monitoring():
    """Test the memory monitoring functionality."""
    logger.info("Testing memory monitoring...")
    
    # Check current memory usage
    memory_info = psutil.virtual_memory()
    logger.info(f"Total memory: {memory_info.total / (1024**3):.1f} GB")
    logger.info(f"Available memory: {memory_info.available / (1024**3):.1f} GB")
    logger.info(f"Memory usage: {memory_info.percent:.1f}%")
    
    # Test the monitoring function
    is_high = check_memory_usage()
    if is_high:
        logger.warning("Memory usage is high!")
    else:
        logger.info("Memory usage is normal")
    
    return True

if __name__ == "__main__":
    test_memory_monitoring() 