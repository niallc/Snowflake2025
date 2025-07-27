# Process All TRMPH Script: Current Design & Usage

## Current Design

### Architecture Overview
The script uses a clean multiprocessing architecture with three main components:

1. **TRMPHProcessor**: Main orchestrator that handles file discovery and processing coordination
2. **ParallelProcessor/SequentialProcessor**: Processing engines that implement the same interface
3. **Worker Functions**: Stateless, independent functions for processing individual files

### Key Design Principles
- **Separation of Concerns**: Processing logic separated from multiprocessing coordination
- **Stateless Workers**: Each worker process is independent with no shared state
- **Error Isolation**: Errors in one worker don't affect others
- **Flexible Processing**: Easy switch between parallel and sequential modes

### File Structure
```
scripts/
├── process_all_trmph.py    # Main entry point with CLI
├── processor.py            # Processing orchestrators
├── workers.py              # Worker functions for multiprocessing
└── config.py               # Configuration management
```

## Important Details

### Worker Process Independence
- Each worker creates its own `BatchProcessor` instance but bypasses resume logic
- Workers use `process_single_file_direct()` to avoid race conditions
- No shared state between worker processes

### File Output Handling
- Uses atomic writes with `atomic_write_pickle_gz()` for thread safety
- Automatic filename uniqueness with counter suffixes
- Output files named: `{original_name}_processed.pkl.gz`

### Data Validation
- Validates `player_to_move` accepts both integers (0,1) and Player enum values
- Checks board shapes, policy dimensions, and metadata completeness
- Comprehensive error reporting per file

## Usage

### Basic Commands
```bash
# Process all files with 6 workers
python scripts/process_all_trmph.py --data-dir data/twoNetGames --output-dir processed_data --max-workers 6

# Sequential processing for debugging
python scripts/process_all_trmph.py --data-dir data/twoNetGames --output-dir processed_data --sequential

# Limit files for testing
python scripts/process_all_trmph.py --data-dir data/twoNetGames --output-dir processed_data --max-files 10 --max-workers 4
```

### Key Parameters
- `--data-dir`: Directory containing .trmph files
- `--output-dir`: Output directory for processed files
- `--max-workers`: Number of parallel processes (default: 6)
- `--sequential`: Force sequential processing
- `--max-files`: Limit number of files to process
- `--position-selector`: Which positions to extract (all/final/penultimate)

### Environment Setup
```bash
source hex_ai_env/bin/activate
export PYTHONPATH=.
```

## Future Refactoring & Cleanup

### 1. Configuration Management
- Move CLI argument parsing to separate module
- Add configuration validation and defaults
- Support configuration files for complex setups

### 2. Progress Reporting
- Add real-time progress bars for long-running jobs
- Implement checkpoint/resume functionality for interrupted runs
- Add estimated time remaining calculations

### 3. Error Handling
- Add retry logic for transient failures
- Implement dead letter queue for failed files
- Add detailed error categorization and reporting

### 4. Performance Optimization
- Add memory usage monitoring and limits
- Implement adaptive worker count based on system resources
- Add batch size optimization for different file types

### 5. Testing Infrastructure
- Add integration tests with sample data
- Implement performance benchmarking
- Add stress testing for large datasets

## Complex Aspects & Testing Needs

### 1. Memory Management
- **Complexity**: Large TRMPH files can consume significant memory
- **Testing**: Monitor memory usage with files of varying sizes
- **Risk**: Out-of-memory errors with very large files

### 2. File System Interactions
- **Complexity**: Atomic writes and filename uniqueness logic
- **Testing**: Test concurrent access scenarios
- **Risk**: Race conditions in file creation

### 3. Data Format Validation
- **Complexity**: Multiple data formats and edge cases
- **Testing**: Test with corrupted or malformed TRMPH files
- **Risk**: Silent failures or incorrect data processing

### 4. Multiprocessing Coordination
- **Complexity**: Process pool management and error propagation
- **Testing**: Test with various worker counts and system loads
- **Risk**: Deadlocks or resource exhaustion

### 5. Error Recovery
- **Complexity**: Handling partial failures in multiprocessing
- **Testing**: Simulate worker process crashes
- **Risk**: Incomplete processing or data loss

## Recommended Testing Scenarios

1. **Small Dataset Test**: 3-5 files with sequential and parallel processing
2. **Large Dataset Test**: 100+ files to test memory and performance
3. **Error Injection Test**: Corrupted files, network failures, disk space issues
4. **Concurrent Access Test**: Multiple script instances on same output directory
5. **Resource Limit Test**: Low memory, low disk space, high CPU load
6. **Interruption Test**: SIGINT handling, graceful shutdown
7. **Format Edge Cases**: Empty files, single-game files, very large games 