# TRMPH Processor Issues and Fixes

## Overview

Analysis of `scripts/process_all_trmph.py` and related functions revealed several issues that could affect large-scale processing jobs. This document outlines the problems and provides specific fixes.

## 🐛 **Critical Issues Found**

### 1. **Statistics Bug: Inconsistent Game Counting**

**Problem**: 
```python
# In process_single_file(), line ~100:
file_stats['games_processed'] += 1  # Incremented for every line
```
The script increments `games_processed` for every line, but invalid lines are skipped without being counted as "processed".

**Impact**: 
- Inconsistent statistics
- Confusing progress reporting
- Potential data quality issues

**Fix**:
```python
# Only increment for successfully processed games
if examples:
    file_stats['valid_games'] += 1
    file_stats['examples_generated'] += len(examples)
    file_stats['games_processed'] += 1  # Move this here
else:
    file_stats['corrupted_games'] += 1
    file_stats['games_processed'] += 1  # Count as processed even if corrupted
```

### 2. **Memory Issue: Large File Processing**

**Problem**: 
```python
# In load_trmph_file():
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()  # Reads entire file into memory
```

**Impact**: 
- Memory exhaustion for very large files
- Potential crashes during processing

**Fix**: Implement streaming processing for large files:
```python
def load_trmph_file_streaming(file_path: str, chunk_size: int = 1000):
    """Load TRMPH file in chunks to avoid memory issues."""
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        for line in f:
            line = line.strip()
            if line:
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk
```

### 3. **File Path Issue: Output File Naming**

**Problem**: 
```python
output_file = self.output_dir / f"{file_path.stem}_processed.pkl.gz"
```

**Impact**: 
- Special characters in filenames could cause issues
- Very long filenames might exceed filesystem limits

**Fix**: Sanitize filenames:
```python
import re
import hashlib

def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """Sanitize filename for safe filesystem operations."""
    # Remove or replace problematic characters
    sanitized = re.sub(r'[^\w\-_.]', '_', filename)
    
    # Truncate if too long
    if len(sanitized) > max_length:
        # Keep extension if present
        name, ext = os.path.splitext(sanitized)
        truncated = name[:max_length-len(ext)-10]  # Leave room for hash
        hash_suffix = hashlib.md5(filename.encode()).hexdigest()[:8]
        sanitized = f"{truncated}_{hash_suffix}{ext}"
    
    return sanitized
```

### 4. **Error Handling Inconsistency**

**Problem**: Different types of errors are handled differently:
- File-level errors: Caught and logged
- Game-level errors: Caught and counted as corrupted
- Processing errors: Caught but not consistently counted

**Fix**: Standardize error handling:
```python
def process_single_file(self, file_path: Path) -> Dict[str, Any]:
    file_stats = {
        'file_path': str(file_path),
        'games_processed': 0,
        'examples_generated': 0,
        'corrupted_games': 0,
        'valid_games': 0,
        'file_errors': 0,  # New field for file-level errors
        'error': None
    }
    
    try:
        games = load_trmph_file(str(file_path))
        
        for i, game_line in enumerate(games):
            try:
                trmph_url, winner = parse_trmph_game_record(game_line)
                examples = extract_training_examples_from_game(trmph_url, winner)
                
                if examples:
                    file_stats['valid_games'] += 1
                    file_stats['examples_generated'] += len(examples)
                else:
                    file_stats['corrupted_games'] += 1
                
                file_stats['games_processed'] += 1
                
            except Exception as e:
                file_stats['corrupted_games'] += 1
                file_stats['games_processed'] += 1
                logger.warning(f"Error processing game {i+1}: {e}")
                
    except Exception as e:
        file_stats['file_errors'] += 1
        file_stats['error'] = str(e)
        logger.error(f"Error processing file {file_path}: {e}")
    
    return file_stats
```

### 5. **Missing Validation: Output Directory Permissions**

**Problem**: No check if output directory is writable.

**Fix**: Add permission validation:
```python
def __init__(self, data_dir: str = "data", output_dir: str = "processed_data"):
    self.data_dir = Path(data_dir)
    self.output_dir = Path(output_dir)
    
    # Validate output directory
    try:
        self.output_dir.mkdir(exist_ok=True)
        # Test write permission
        test_file = self.output_dir / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
    except (PermissionError, OSError) as e:
        raise ValueError(f"Output directory {output_dir} is not writable: {e}")
```

### 6. **Potential Race Condition: Concurrent Processing**

**Problem**: Multiple processes writing to same output directory.

**Fix**: Add unique identifiers to output files:
```python
import time
import uuid

def get_unique_output_filename(self, original_filename: str) -> str:
    """Generate unique output filename to avoid conflicts."""
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    sanitized_name = sanitize_filename(original_filename)
    return f"{sanitized_name}_{timestamp}_{unique_id}_processed.pkl.gz"
```

## 🧪 **Test Improvements**

### 1. **Add Performance Tests**
```python
def test_memory_usage_large_files(self):
    """Test memory usage with large files."""
    import psutil
    import os
    
    # Create large file
    large_content = "\n".join([f"#13,a1b2c3 1" for _ in range(10000)])
    self.create_test_trmph_file("large.trmph", large_content)
    
    # Monitor memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
    processor.process_single_file(self.data_dir / "large.trmph")
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (< 100MB)
    assert memory_increase < 100 * 1024 * 1024
```

### 2. **Add Concurrency Tests**
```python
def test_concurrent_file_processing(self):
    """Test that multiple processes can run simultaneously."""
    import multiprocessing as mp
    
    # Create multiple files
    for i in range(10):
        self.create_test_trmph_file(f"concurrent{i}.trmph", "#13,a1b2c3 1\n")
    
    def process_file(file_path):
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        return processor.process_single_file(file_path)
    
    # Process files in parallel
    with mp.Pool(4) as pool:
        file_paths = list(self.data_dir.glob("*.trmph"))
        results = pool.map(process_file, file_paths)
    
    # All should succeed
    assert all(r['error'] is None for r in results)
```

### 3. **Add Data Integrity Tests**
```python
def test_output_data_integrity(self):
    """Test that output data is consistent and valid."""
    content = "#13,a1b2c3 1\n#13,a1b2 2\n"
    self.create_test_trmph_file("integrity_test.trmph", content)
    
    processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
    processor.process_single_file(self.data_dir / "integrity_test.trmph")
    
    # Load and validate output
    output_files = list(self.output_dir.glob("*_processed.pkl.gz"))
    assert len(output_files) == 1
    
    with gzip.open(output_files[0], 'rb') as f:
        data = pickle.load(f)
    
    # Validate data structure
    assert 'examples' in data
    assert 'source_file' in data
    assert 'processing_stats' in data
    assert 'processed_at' in data
    
    # Validate examples
    for example in data['examples']:
        assert 'board' in example
        assert 'policy' in example
        assert 'value' in example
        assert 'metadata' in example
        
        # Validate board shape
        assert example['board'].shape == (2, 13, 13)
        
        # Validate policy shape
        if example['policy'] is not None:
            assert example['policy'].shape == (169,)
```

## 🚀 **Recommended Implementation Priority**

### High Priority (Fix Before Large Job)
1. **Statistics Bug Fix** - Critical for accurate reporting
2. **Error Handling Standardization** - Important for reliability
3. **Output Directory Validation** - Prevents job failures

### Medium Priority
4. **Filename Sanitization** - Prevents edge case failures
5. **Memory Usage Monitoring** - Important for large files

### Low Priority
6. **Concurrency Improvements** - Nice to have for parallel processing

## 📊 **Monitoring Recommendations**

### 1. **Add Progress Monitoring**
```python
def log_progress(self, current_file: int, total_files: int, stats: Dict):
    """Log detailed progress information."""
    elapsed = time.time() - self.stats['start_time']
    rate = current_file / elapsed if elapsed > 0 else 0
    eta = (total_files - current_file) / rate if rate > 0 else 0
    
    logger.info(f"Progress: {current_file}/{total_files} ({current_file/total_files*100:.1f}%)")
    logger.info(f"Rate: {rate:.2f} files/sec, ETA: {eta/60:.1f} minutes")
    logger.info(f"Valid games: {stats['valid_games']}, Corrupted: {stats['corrupted_games']}")
```

### 2. **Add Memory Monitoring**
```python
def log_memory_usage(self):
    """Log current memory usage."""
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage: {memory_mb:.1f} MB")
```

### 3. **Add Data Quality Metrics**
```python
def calculate_data_quality_metrics(self, examples: List[Dict]) -> Dict:
    """Calculate data quality metrics."""
    total_examples = len(examples)
    examples_with_policy = sum(1 for e in examples if e['policy'] is not None)
    examples_without_policy = total_examples - examples_with_policy
    
    return {
        'total_examples': total_examples,
        'examples_with_policy': examples_with_policy,
        'examples_without_policy': examples_without_policy,
        'policy_coverage': examples_with_policy / total_examples if total_examples > 0 else 0
    }
```

## Conclusion

The TRMPH processor is generally well-designed but has several issues that should be addressed before running large-scale processing jobs. The most critical fixes are the statistics bug and error handling standardization. The additional tests and monitoring will help ensure reliable processing of large datasets. 