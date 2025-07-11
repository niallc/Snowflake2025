#!/usr/bin/env python3
"""
Standalone script to test system analysis functionality.
"""

import torch
import psutil
import platform
import subprocess
from typing import Dict, Tuple

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
        'gpu_available': torch.cuda.is_available(),
    }
    
    if info['gpu_available']:
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / (1024**3)
        info['gpu_memory_cached'] = torch.cuda.memory_reserved(0) / (1024**3)
    
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
            print(f"Could not get memory compression info: {e}")
    
    return info

def calculate_optimal_shard_size(target_memory_gb: float = 8.0, 
                               games_per_shard: int = 1000) -> Tuple[int, Dict]:
    """Calculate optimal shard size based on available memory."""
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
        print(f"  GPU Memory: {system_info['gpu_memory_total']:.1f} GB")
    
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