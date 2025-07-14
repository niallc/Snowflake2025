#!/usr/bin/env python3
"""
Resource monitoring script for Hex AI training.
Monitors CPU, GPU, memory, and training-specific metrics.
"""

import psutil
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

def get_system_info():
    """Get basic system information."""
    info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'gpu_available': torch.cuda.is_available() or torch.backends.mps.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    }
    return info

def get_current_utilization():
    """Get current resource utilization."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    # Get GPU info if available
    gpu_info = {}
    if torch.cuda.is_available():
        try:
            gpu_info = {
                'gpu_memory_used_mb': torch.cuda.memory_allocated() / (1024**2),
                'gpu_memory_total_mb': torch.cuda.get_device_properties(0).total_memory / (1024**2),
                'gpu_memory_percent': (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
            }
        except Exception as e:
            gpu_info = {'error': str(e)}
    elif torch.backends.mps.is_available():
        # MPS doesn't provide detailed memory info, but we can check if it's being used
        gpu_info = {'device': 'mps', 'note': 'MPS memory info not available'}
    
    return {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_used_gb': memory.used / (1024**3),
        'memory_available_gb': memory.available / (1024**3),
        'gpu_info': gpu_info
    }

def get_python_processes():
    """Get information about Python processes."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_percent': proc.info['memory_percent'],
                    'memory_mb': proc.info['memory_info'].rss / (1024**2)
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes

def analyze_training_bottlenecks():
    """Analyze potential training bottlenecks."""
    bottlenecks = []
    
    # Check CPU utilization
    cpu_percent = psutil.cpu_percent(interval=2)
    if cpu_percent < 50:
        bottlenecks.append(f"CPU underutilized: {cpu_percent:.1f}% (consider increasing batch size or using more workers)")
    elif cpu_percent > 90:
        bottlenecks.append(f"CPU bottleneck: {cpu_percent:.1f}% (consider reducing batch size or workers)")
    
    # Check memory utilization
    memory = psutil.virtual_memory()
    if memory.percent < 50:
        bottlenecks.append(f"Memory underutilized: {memory.percent:.1f}% (consider increasing batch size)")
    elif memory.percent > 90:
        bottlenecks.append(f"Memory bottleneck: {memory.percent:.1f}% (consider reducing batch size)")
    
    # Check GPU utilization
    if torch.cuda.is_available():
        gpu_memory_percent = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
        if gpu_memory_percent < 50:
            bottlenecks.append(f"GPU memory underutilized: {gpu_memory_percent:.1f}% (consider increasing batch size)")
        elif gpu_memory_percent > 90:
            bottlenecks.append(f"GPU memory bottleneck: {gpu_memory_percent:.1f}% (consider reducing batch size)")
    
    return bottlenecks

def monitor_continuously(duration_minutes=5, interval_seconds=10):
    """Monitor resources continuously."""
    print(f"Monitoring resources for {duration_minutes} minutes (every {interval_seconds}s)")
    print("="*60)
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    # Get system info
    system_info = get_system_info()
    print(f"System: {system_info['cpu_count']} cores, {system_info['memory_total_gb']:.1f}GB RAM")
    print(f"Device: {system_info['device']}")
    print()
    
    while time.time() < end_time:
        utilization = get_current_utilization()
        processes = get_python_processes()
        bottlenecks = analyze_training_bottlenecks()
        
        print(f"[{utilization['timestamp']}]")
        print(f"CPU: {utilization['cpu_percent']:.1f}% | Memory: {utilization['memory_percent']:.1f}% ({utilization['memory_used_gb']:.1f}GB)")
        
        if utilization['gpu_info']:
            if 'gpu_memory_percent' in utilization['gpu_info']:
                print(f"GPU Memory: {utilization['gpu_info']['gpu_memory_percent']:.1f}% ({utilization['gpu_info']['gpu_memory_used_mb']:.1f}MB)")
            else:
                print(f"GPU: {utilization['gpu_info']}")
        
        if processes:
            print("Python processes:")
            for proc in processes:
                print(f"  PID {proc['pid']}: {proc['name']} - CPU: {proc['cpu_percent']:.1f}%, Memory: {proc['memory_mb']:.1f}MB")
        
        if bottlenecks:
            print("Bottlenecks detected:")
            for bottleneck in bottlenecks:
                print(f"  ⚠️  {bottleneck}")
        
        print("-" * 40)
        time.sleep(interval_seconds)

def get_parallel_training_recommendations():
    """Get recommendations for parallel training."""
    system_info = get_system_info()
    utilization = get_current_utilization()
    
    recommendations = []
    
    # Analyze current utilization
    if utilization['cpu_percent'] < 30:
        recommendations.append("Low CPU utilization - good candidate for parallel training")
    if utilization['memory_percent'] < 50:
        recommendations.append("Low memory utilization - can run multiple processes")
    
    # Calculate optimal number of parallel processes
    cpu_cores = system_info['cpu_count']
    memory_gb = system_info['memory_total_gb']
    
    # Conservative estimate: 2GB per process, leave 4GB for system
    max_memory_processes = max(1, int((memory_gb - 4) / 2))
    
    # CPU-based estimate: leave 2 cores for system
    max_cpu_processes = max(1, cpu_cores - 2)
    
    optimal_processes = min(max_memory_processes, max_cpu_processes)
    
    recommendations.append(f"Recommended parallel processes: {optimal_processes}")
    recommendations.append(f"  - Memory-based: {max_memory_processes} processes")
    recommendations.append(f"  - CPU-based: {max_cpu_processes} processes")
    
    return recommendations

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor system resources for Hex AI training')
    parser.add_argument('--duration', type=int, default=5, help='Monitoring duration in minutes')
    parser.add_argument('--interval', type=int, default=10, help='Update interval in seconds')
    parser.add_argument('--recommendations', action='store_true', help='Show parallel training recommendations')
    
    args = parser.parse_args()
    
    if args.recommendations:
        print("PARALLEL TRAINING RECOMMENDATIONS")
        print("="*50)
        recommendations = get_parallel_training_recommendations()
        for rec in recommendations:
            print(f"• {rec}")
        print()
    
    monitor_continuously(args.duration, args.interval) 