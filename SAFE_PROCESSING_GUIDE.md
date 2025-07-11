# Safe Processing Guide

## Overview

The `safe_shard_processing.py` script provides a memory-safe way to process your trmph data files into sharded pickle files for training. It includes:

- **Memory monitoring**: Exits if memory usage exceeds 80%
- **Emergency shutdown**: Graceful handling of Ctrl+C and system signals
- **Reduced parallelism**: Uses 4 workers instead of 8 to reduce memory pressure
- **Incremental processing**: Reads files one at a time until enough games for a shard

## Safety Features

### Memory Monitoring
- Checks memory usage every 10 games processed
- Exits gracefully if memory usage exceeds 80%
- Logs memory usage for monitoring

### Process Management
- Reduced worker count from 8 to 4
- Proper cleanup of child processes
- Signal handlers for graceful shutdown

### Emergency Shutdown
- Ctrl+C triggers graceful shutdown
- System signals (SIGTERM) handled properly
- Logs shutdown status

## Usage

### 1. Test Memory Monitoring
```bash
python test_memory_monitoring.py
```

### 2. Test with Small Dataset
```bash
python test_safe_processing.py
```

### 3. Run Full Processing
```bash
python safe_shard_processing.py
```

## Configuration

You can adjust these safety parameters in `safe_shard_processing.py`:

```python
MAX_MEMORY_PERCENT = 80  # Exit if memory usage exceeds 80%
MEMORY_CHECK_INTERVAL = 10  # Check memory every 10 games processed
MAX_WORKERS = 4  # Reduced from 8 to 4
```

## Monitoring

The script provides detailed logging:
- Memory usage checks
- Progress through files and shards
- Processing statistics
- Emergency shutdown notifications

## Emergency Procedures

If the script triggers emergency shutdown:

1. **Check logs**: Look for memory usage warnings
2. **Clean up processes**: The script should clean up automatically, but you can check with:
   ```bash
   ps aux | grep python | grep -v grep
   ```
3. **Kill orphaned processes**: If needed:
   ```bash
   pkill -f "python.*safe_shard_processing"
   ```

## Expected Behavior

- **Normal operation**: Processes files sequentially, creates shards of 1000 games each
- **Memory pressure**: Logs warnings and exits gracefully
- **Interruption**: Handles Ctrl+C and system signals properly
- **Completion**: Creates numbered shard files in `data/processed/`

## Performance

- **Slower than original**: Due to reduced parallelism and memory checks
- **Much safer**: Won't crash your system
- **Resumable**: Can restart from where it left off (shards are numbered)

## Troubleshooting

### High Memory Usage
- Reduce `MAX_WORKERS` to 2
- Increase `MEMORY_CHECK_INTERVAL` to 20
- Lower `MAX_MEMORY_PERCENT` to 70

### Orphaned Processes
- Check with `ps aux | grep python`
- Kill with `pkill -f "python.*ProcessPoolExecutor"`

### Corrupted Shards
- Delete partial shards and restart
- Check logs for specific errors 