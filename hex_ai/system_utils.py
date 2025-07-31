"""
System utilities for determining optimal training parameters.

This module helps determine:
- Available system memory
- Optimal batch sizes
- Optimal shard sizes
- GPU capabilities
"""

import torch
import psutil
import platform
import subprocess
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def check_virtual_env(expected_env="hex_ai_env"):
    """
    DEPRECATED: Environment validation is now handled automatically in hex_ai/__init__.py
    
    This function is kept for backward compatibility but does nothing.
    Environment validation happens automatically when importing hex_ai.
    
    Args:
        expected_env: Name of the expected virtual environment (ignored)
    """
    # Environment validation is now handled in hex_ai/__init__.py
    # This function is kept for backward compatibility
    pass


def get_system_info() -> Dict:
    """Get comprehensive system information."""
    info = {
        'platform': platform.system(),
        'architecture': platform.machine(),
        'cpu_count': psutil.cpu_count(),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'memory_percent_used': psutil.virtual_memory().percent,
        'gpu_available': torch.cuda.is_available() or torch.backends.mps.is_available(),
    }
    
    if info['gpu_available']:
        if torch.cuda.is_available():
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / (1024**3)
            info['gpu_memory_cached'] = torch.cuda.memory_reserved(0) / (1024**3)
        elif torch.backends.mps.is_available():
            info['gpu_count'] = 1  # MPS typically has 1 GPU
            info['gpu_memory_total'] = None  # MPS doesn't expose memory info
            info['gpu_memory_allocated'] = None
            info['gpu_memory_cached'] = None
    
    return info


def get_memory_compression_info() -> Dict:
    """Get memory compression information (macOS specific)."""
    info = {}
    
    if platform.system() == 'Darwin':
        try:
            # Check memory pressure
            result = subprocess.run(['vm_stat'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Pages occupied by compressor' in line:
                        compressed_pages = int(line.split(':')[1].strip())
                        compressed_mb = compressed_pages * 4096 / (1024**2)
                        info['memory_compressed_mb'] = compressed_mb
                        break
        except Exception as e:
            logger.warning(f"Could not get memory compression info: {e}")
    
    return info


def calculate_optimal_shard_size(target_memory_gb: float = 8.0, 
                               games_per_shard: int = 1000) -> Tuple[int, Dict]:
    """
    Calculate optimal shard size based on available memory.
    
    Args:
        target_memory_gb: Target memory usage for data loading
        games_per_shard: Current games per shard
        
    Returns:
        optimal_games_per_shard: Recommended games per shard
        analysis: Detailed analysis of memory usage
    """
    system_info = get_system_info()
    available_memory_gb = system_info['memory_available_gb']
    
    # Estimate memory per game in processed format
    # Based on current observations: ~328KB per 1000 games
    memory_per_game_mb = 0.328  # 328KB per 1000 games
    
    # Calculate how many games we can fit in target memory
    max_games_in_memory = int((target_memory_gb * 1024) / memory_per_game_mb)
    
    # Round down to nearest multiple of current shard size
    optimal_games_per_shard = (max_games_in_memory // games_per_shard) * games_per_shard
    
    # Ensure we don't exceed available memory
    if optimal_games_per_shard * memory_per_game_mb > available_memory_gb * 1024:
        optimal_games_per_shard = int((available_memory_gb * 1024 * 0.8) / memory_per_game_mb)
        optimal_games_per_shard = (optimal_games_per_shard // games_per_shard) * games_per_shard
    
    analysis = {
        'available_memory_gb': available_memory_gb,
        'target_memory_gb': target_memory_gb,
        'memory_per_game_mb': memory_per_game_mb,
        'max_games_in_memory': max_games_in_memory,
        'current_games_per_shard': games_per_shard,
        'optimal_games_per_shard': optimal_games_per_shard,
        'estimated_shards_in_memory': optimal_games_per_shard // games_per_shard,
        'memory_usage_percent': (optimal_games_per_shard * memory_per_game_mb) / (available_memory_gb * 1024) * 100
    }
    
    return optimal_games_per_shard, analysis


def calculate_optimal_batch_size(model_memory_mb: float = 50.0,
                               target_memory_gb: float = 4.0) -> Tuple[int, Dict]:
    """
    Calculate optimal batch size based on model and memory constraints.
    
    Args:
        model_memory_mb: Estimated memory per model forward/backward pass
        target_memory_gb: Target memory usage for training
        
    Returns:
        optimal_batch_size: Recommended batch size
        analysis: Detailed analysis
    """
    system_info = get_system_info()
    available_memory_gb = system_info['memory_available_gb']
    
    # Estimate memory per sample (board + policy + value tensors)
    # Board: 11x11x3 = 363 floats = 1.45KB
    # Policy: 121 floats = 0.48KB  
    # Value: 1 float = 0.004KB
    # Total per sample: ~2KB
    memory_per_sample_mb = 0.002
    
    # Calculate max batch size based on available memory
    max_batch_size = int((target_memory_gb * 1024) / (model_memory_mb + memory_per_sample_mb))
    
    # Ensure we don't exceed available memory
    if max_batch_size * memory_per_sample_mb > available_memory_gb * 1024 * 0.8:
        max_batch_size = int((available_memory_gb * 1024 * 0.8) / memory_per_sample_mb)
    
    # Round to power of 2 for efficiency
    optimal_batch_size = 2 ** (max_batch_size.bit_length() - 1)
    
    analysis = {
        'available_memory_gb': available_memory_gb,
        'target_memory_gb': target_memory_gb,
        'model_memory_mb': model_memory_mb,
        'memory_per_sample_mb': memory_per_sample_mb,
        'max_batch_size': max_batch_size,
        'optimal_batch_size': optimal_batch_size,
        'estimated_memory_usage_gb': (optimal_batch_size * memory_per_sample_mb + model_memory_mb) / 1024
    }
    
    return optimal_batch_size, analysis


def print_system_analysis():
    """Print comprehensive system analysis for training optimization."""
    print("=== System Analysis for Hex AI Training ===")
    
    # System info
    system_info = get_system_info()
    print(f"\nSystem Information:")
    print(f"  Platform: {system_info['platform']}")
    print(f"  CPU Cores: {system_info['cpu_count']} physical, {system_info['cpu_count_logical']} logical")
    print(f"  Total Memory: {system_info['memory_total_gb']:.1f} GB")
    print(f"  Available Memory: {system_info['memory_available_gb']:.1f} GB")
    print(f"  Memory Usage: {system_info['memory_percent_used']:.1f}%")
    print(f"  GPU Available: {system_info['gpu_available']}")
    
    if system_info['gpu_available']:
        print(f"  GPU Count: {system_info['gpu_count']}")
        if system_info['gpu_memory_total'] is not None:
            print(f"  GPU Memory: {system_info['gpu_memory_total']:.1f} GB")
        else:
            print(f"  GPU Type: Apple MPS (memory info not available)")
    
    # Memory compression (macOS)
    compression_info = get_memory_compression_info()
    if compression_info:
        print(f"  Memory Compressed: {compression_info.get('memory_compressed_mb', 0):.1f} MB")
    
    # Shard size analysis
    print(f"\nShard Size Analysis:")
    optimal_shard_size, shard_analysis = calculate_optimal_shard_size()
    print(f"  Current Games per Shard: {shard_analysis['current_games_per_shard']}")
    print(f"  Optimal Games per Shard: {shard_analysis['optimal_games_per_shard']}")
    print(f"  Estimated Shards in Memory: {shard_analysis['estimated_shards_in_memory']}")
    print(f"  Memory Usage: {shard_analysis['memory_usage_percent']:.1f}%")
    
    # Batch size analysis
    print(f"\nBatch Size Analysis:")
    optimal_batch_size, batch_analysis = calculate_optimal_batch_size()
    print(f"  Optimal Batch Size: {batch_analysis['optimal_batch_size']}")
    print(f"  Estimated Memory Usage: {batch_analysis['estimated_memory_usage_gb']:.2f} GB")
    
    # Recommendations
    print(f"\nRecommendations:")
    if system_info['memory_available_gb'] > 16:
        print(f"  ✅ High memory system - can load multiple shards simultaneously")
    elif system_info['memory_available_gb'] > 8:
        print(f"  ⚠️  Moderate memory - consider streaming for large datasets")
    else:
        print(f"  ❌ Low memory - use streaming approach for large datasets")
    
    if system_info['gpu_available']:
        print(f"  ✅ GPU available - consider mixed precision training")
    else:
        print(f"  ⚠️  CPU-only training - consider cloud GPU for large scale")


if __name__ == "__main__":
    print_system_analysis() 