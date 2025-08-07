"""
GPU monitoring utilities for performance optimization and debugging.
"""

import torch
import logging
from typing import Dict, Optional, Any
import psutil
import time

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> Optional[Dict[str, Any]]:
    """
    Get detailed GPU memory usage information.
    
    Returns:
        Dictionary with GPU memory information, or None if no GPU available
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        # Get GPU device properties
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        # Get memory usage
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        total = props.total_memory
        
        # Convert to MB for readability
        allocated_mb = allocated / (1024 * 1024)
        reserved_mb = reserved / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        free_mb = total_mb - allocated_mb
        
        return {
            'device_name': props.name,
            'device_id': device,
            'total_mb': total_mb,
            'allocated_mb': allocated_mb,
            'reserved_mb': reserved_mb,
            'free_mb': free_mb,
            'utilization_percent': (allocated_mb / total_mb) * 100,
            'total_bytes': total,
            'allocated_bytes': allocated,
            'reserved_bytes': reserved
        }
    except Exception as e:
        logger.error(f"Failed to get GPU memory info: {e}")
        return None


def get_system_memory_info() -> Dict[str, Any]:
    """
    Get system memory usage information.
    
    Returns:
        Dictionary with system memory information
    """
    memory = psutil.virtual_memory()
    return {
        'total_mb': memory.total / (1024 * 1024),
        'available_mb': memory.available / (1024 * 1024),
        'used_mb': memory.used / (1024 * 1024),
        'percent_used': memory.percent,
        'free_mb': memory.free / (1024 * 1024)
    }


def log_memory_status(prefix: str = ""):
    """
    Log current memory status for debugging.
    
    Args:
        prefix: Optional prefix for log messages
    """
    # System memory
    sys_mem = get_system_memory_info()
    logger.info(f"{prefix}System Memory: {sys_mem['used_mb']:.1f}MB used / "
               f"{sys_mem['total_mb']:.1f}MB total ({sys_mem['percent_used']:.1f}%)")
    
    # GPU memory
    gpu_mem = get_gpu_memory_info()
    if gpu_mem:
        logger.info(f"{prefix}GPU Memory: {gpu_mem['allocated_mb']:.1f}MB allocated / "
                   f"{gpu_mem['total_mb']:.1f}MB total ({gpu_mem['utilization_percent']:.1f}%)")
    else:
        logger.info(f"{prefix}GPU Memory: Not available")


def estimate_optimal_batch_size(model_input_size: int, max_batch_size: int = 256) -> int:
    """
    Estimate optimal batch size based on available GPU memory.
    
    Args:
        model_input_size: Size of a single model input in bytes
        max_batch_size: Maximum batch size to consider
        
    Returns:
        Estimated optimal batch size
    """
    gpu_mem = get_gpu_memory_info()
    if not gpu_mem:
        logger.warning("No GPU available, using default batch size of 32")
        return 32
    
    # Estimate memory needed for batch processing
    # Assume we need 2x the input size for intermediate computations
    memory_per_sample = model_input_size * 2
    
    # Use 80% of available GPU memory to be safe
    available_memory = gpu_mem['free_mb'] * 1024 * 1024 * 0.8
    
    estimated_batch_size = int(available_memory / memory_per_sample)
    
    # Clamp to reasonable bounds
    optimal_batch_size = max(1, min(estimated_batch_size, max_batch_size))
    
    logger.info(f"GPU Memory Analysis:")
    logger.info(f"  Available GPU memory: {gpu_mem['free_mb']:.1f}MB")
    logger.info(f"  Memory per sample: {memory_per_sample / (1024*1024):.1f}MB")
    logger.info(f"  Estimated batch size: {estimated_batch_size}")
    logger.info(f"  Optimal batch size: {optimal_batch_size}")
    
    return optimal_batch_size


def monitor_gpu_usage(duration: float = 10.0, interval: float = 1.0):
    """
    Monitor GPU usage over time for debugging.
    
    Args:
        duration: Total monitoring duration in seconds
        interval: Monitoring interval in seconds
    """
    logger.info(f"Starting GPU monitoring for {duration}s (interval: {interval}s)")
    
    start_time = time.time()
    samples = []
    
    while time.time() - start_time < duration:
        gpu_mem = get_gpu_memory_info()
        if gpu_mem:
            samples.append({
                'timestamp': time.time() - start_time,
                'allocated_mb': gpu_mem['allocated_mb'],
                'utilization_percent': gpu_mem['utilization_percent']
            })
        
        time.sleep(interval)
    
    # Log summary
    if samples:
        allocated_values = [s['allocated_mb'] for s in samples]
        utilization_values = [s['utilization_percent'] for s in samples]
        
        logger.info(f"GPU Monitoring Summary ({len(samples)} samples):")
        logger.info(f"  Average allocated: {sum(allocated_values)/len(allocated_values):.1f}MB")
        logger.info(f"  Peak allocated: {max(allocated_values):.1f}MB")
        logger.info(f"  Average utilization: {sum(utilization_values)/len(utilization_values):.1f}%")
        logger.info(f"  Peak utilization: {max(utilization_values):.1f}%")


def clear_gpu_cache():
    """Clear GPU cache and log memory status."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
        
        # Log memory status after clearing
        gpu_mem = get_gpu_memory_info()
        if gpu_mem:
            logger.info(f"GPU Memory after cache clear: {gpu_mem['allocated_mb']:.1f}MB allocated")
    else:
        logger.info("No GPU available, skipping cache clear")
