#!/usr/bin/env python3
"""
Test script to verify MCTS performance improvements.
Tests ModelWrapper caching and timing instrumentation.
"""

import time
import requests
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_mcts_performance():
    """Test MCTS performance with and without caching."""
    
    # Test parameters
    base_url = "http://localhost:5001"
    test_trmph = "#13,"  # Empty board in correct TRMPH format
    model_id = "model1"
    
    print("=== MCTS Performance Test ===")
    print(f"Testing with model: {model_id}")
    print(f"Base URL: {base_url}")
    print()
    
    # Test 1: First call (should create ModelWrapper)
    print("Test 1: First MCTS call (ModelWrapper creation)")
    start_time = time.time()
    
    try:
        response = requests.post(f"{base_url}/api/mcts_move", json={
            "trmph": test_trmph,
            "model_id": model_id,
            "num_simulations": 50,  # Reduced for faster testing
            "temperature": 1.0
        }, timeout=30)
        
        first_call_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ First call completed in {first_call_time:.3f}s")
            print(f"  Success: {result.get('success', False)}")
            print(f"  Move made: {result.get('move_made', 'None')}")
        else:
            print(f"✗ First call failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ First call error: {e}")
        return False
    
    print()
    
    # Test 2: Second call (should use cached ModelWrapper)
    print("Test 2: Second MCTS call (cached ModelWrapper)")
    start_time = time.time()
    
    try:
        response = requests.post(f"{base_url}/api/mcts_move", json={
            "trmph": test_trmph,
            "model_id": model_id,
            "num_simulations": 50,
            "temperature": 1.0
        }, timeout=30)
        
        second_call_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Second call completed in {second_call_time:.3f}s")
            print(f"  Success: {result.get('success', False)}")
            print(f"  Move made: {result.get('move_made', 'None')}")
        else:
            print(f"✗ Second call failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Second call error: {e}")
        return False
    
    print()
    
    # Performance analysis
    speedup = first_call_time / second_call_time if second_call_time > 0 else 0
    time_saved = first_call_time - second_call_time
    
    print("=== Performance Analysis ===")
    print(f"First call time:  {first_call_time:.3f}s")
    print(f"Second call time: {second_call_time:.3f}s")
    print(f"Time saved:       {time_saved:.3f}s")
    print(f"Speedup:          {speedup:.1f}x")
    
    if speedup > 2.0:
        print("✓ Significant performance improvement detected!")
        return True
    else:
        print("⚠ Performance improvement minimal or not detected")
        return False

def test_cache_status():
    """Test cache status endpoint."""
    print("\n=== Cache Status Test ===")
    
    try:
        response = requests.get("http://localhost:5001/api/cache/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✓ Cache status: {status}")
            return True
        else:
            print(f"✗ Cache status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cache status error: {e}")
        return False

if __name__ == "__main__":
    print("MCTS Performance Test")
    print("Make sure the Flask server is running on localhost:5001")
    print()
    
    # Test cache status
    cache_ok = test_cache_status()
    
    # Test performance
    perf_ok = test_mcts_performance()
    
    print("\n=== Test Summary ===")
    print(f"Cache status: {'✓ PASS' if cache_ok else '✗ FAIL'}")
    print(f"Performance:  {'✓ PASS' if perf_ok else '✗ FAIL'}")
    
    if cache_ok and perf_ok:
        print("\n🎉 All tests passed! ModelWrapper caching is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the server logs for details.")
        sys.exit(1)
