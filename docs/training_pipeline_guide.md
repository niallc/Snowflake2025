# Training Pipeline Guide

## Overview

The new training pipeline (`scripts/training_pipeline.py`) replaces the previous bash-based approach with a comprehensive Python pipeline that provides:

- **Multi-worker self-play generation** (2-5 workers as recommended)
- **Modular step execution** (can run individual steps or combinations)
- **Better error handling and progress tracking**
- **Automatic cleanup of intermediate files**
- **Comprehensive logging and monitoring**

## Architecture

### Pipeline Steps

1. **Self-Play Generation** (`SelfPlayStep`)
   - Multi-worker parallel generation
   - Each worker runs independently with different seeds
   - Streaming save to avoid data loss
   - Configurable search widths and temperature

2. **Data Preprocessing** (`PreprocessingStep`)
   - Combines multiple .trmph files
   - Removes duplicate games
   - Splits into manageable chunks

3. **TRMPH Processing** (`TRMPHProcessingStep`)
   - Converts .trmph files to training positions
   - Configurable position selection (all, final, penultimate)
   - Multi-worker processing for speed

4. **Data Shuffling** (`ShufflingStep`)
   - Shuffles processed data to prevent value head fingerprinting
   - Uses bucket-based approach for large datasets
   - Resume capability for interrupted runs

5. **Model Training** (`TrainingStep`)
   - Hyperparameter tuning with multiple data sources
   - Automatic experiment naming and organization
   - Resume capability from checkpoints

### Key Components

- **`PipelineConfig`**: Centralized configuration management
- **`TrainingPipeline`**: Main orchestrator class
- **Step Classes**: Each processing step as a separate class
- **Progress Tracking**: Resume capability and status reporting

## Usage

### Basic Usage

```bash
# Run complete pipeline
python scripts/training_pipeline.py --model-path checkpoints/experiment/epoch4_mini1.pt.gz

# Run with custom settings
python scripts/training_pipeline.py \
  --model-path checkpoints/experiment/epoch4_mini1.pt.gz \
  --num-games 50000 \
  --num-workers 5 \
  --temperature 1.0
```

### Selective Execution

```bash
# Run only self-play and preprocessing
python scripts/training_pipeline.py \
  --model-path checkpoints/experiment/epoch4_mini1.pt.gz \
  --no-trmph-processing \
  --no-shuffling \
  --no-training

# Run only training (using existing processed data)
python scripts/training_pipeline.py \
  --model-path checkpoints/experiment/epoch4_mini1.pt.gz \
  --no-selfplay \
  --no-preprocessing \
  --no-trmph-processing \
  --no-shuffling

# Use existing self-play data (for flexible interruption/resume)
python scripts/training_pipeline.py \
  --model-path checkpoints/experiment/epoch4_mini1.pt.gz \
  --selfplay-dir data/sf25/selfplay_20250805_143022 \
  --no-selfplay

# Process existing self-play data through to training
python scripts/training_pipeline.py \
  --model-path checkpoints/experiment/epoch4_mini1.pt.gz \
  --selfplay-dir data/sf25/selfplay_20250805_143022
```

### Advanced Configuration

```bash
# Custom search parameters
python scripts/training_pipeline.py \
  --model-path checkpoints/experiment/epoch4_mini1.pt.gz \
  --search-widths 15 10 5 \
  --temperature 1.2 \
  --batch-size 256

# Custom data processing
python scripts/training_pipeline.py \
  --model-path checkpoints/experiment/epoch4_mini1.pt.gz \
  --chunk-size 20000 \
  --position-selector final \
  --max-workers-trmph 8 \
  --num-buckets-shuffle 200
```

## Configuration Options

### Model Configuration
- `--model-path`: Path to model checkpoint directory
- `--model-epoch`: Model epoch number (default: 4)
- `--model-mini`: Model mini-epoch number (default: 1)

### Self-Play Configuration
- `--num-games`: Number of games to generate (default: 100000)
- `--num-workers`: Number of self-play workers (default: 3)
- `--search-widths`: Search widths for minimax (default: [13, 8])
- `--temperature`: Temperature for move sampling (default: 1.5)
- `--batch-size`: Batch size for inference (default: 128)
- `--cache-size`: Cache size for model inference (default: 60000)

### Data Configuration
- `--base-data-dir`: Base directory for data (default: "data")
- `--existing-shuffled-data-dir`: Existing shuffled data directory (default: "data/processed/shuffled")
- `--selfplay-dir`: Use existing self-play directory (skip self-play generation)
- `--chunk-size`: Chunk size for preprocessing (default: 10000)
- `--position-selector`: Position selector for TRMPH processing (default: "all")
- `--max-workers-trmph`: Max workers for TRMPH processing (default: 6)
- `--num-buckets-shuffle`: Number of buckets for shuffling (default: 100)

### Training Configuration
- `--max-samples`: Max training samples (default: 35000000)
- `--results-dir`: Results directory (default: "checkpoints/hyperparameter_tuning")
- `--skip-files`: Files to skip from each data directory (default: [0, 0, 200])

### Pipeline Control
- `--no-selfplay`: Skip self-play step
- `--no-preprocessing`: Skip preprocessing step
- `--no-trmph-processing`: Skip TRMPH processing step
- `--no-shuffling`: Skip shuffling step
- `--no-training`: Skip training step
- `--no-cleanup`: Keep intermediate files

## Data Flow

### Directory Structure

```
data/
├── sf25/
│   └── selfplay_YYYYMMDD_HHMMSS/          # Self-play output
│       ├── worker_0/
│       ├── worker_1/
│       └── ...
├── cleaned/
│   └── run_YYYYMMDD_HHMMSS/               # Preprocessed data
├── processed/
│   ├── run_YYYYMMDD_HHMMSS/               # TRMPH processed data
│   ├── shuffled_YYYYMMDD_HHMMSS/          # Shuffled data
│   └── temp_buckets_YYYYMMDD_HHMMSS/      # Temporary files
└── processed/
    └── shuffled/                          # Existing data (for training)
```

### Data Mixing Strategy

The pipeline automatically handles data mixing for training:

1. **New data**: Generated from self-play in current run
2. **Existing data**: Previously processed and shuffled data
3. **Training mix**: Combines both sources with configurable skip patterns

This allows for:
- Incremental training on new data
- Maintaining training on historical data
- Avoiding reprocessing of existing data

## Multi-Worker Self-Play

### Design

The multi-worker system spawns independent processes that:
- Run with different random seeds
- Generate games in parallel
- Save results to separate directories
- Combine automatically for downstream processing

### Flexible Self-Play with Interruption/Resume

For scenarios where you want to generate games with flexible interruption and resume (like your original workflow), you have two options:

#### Option 1: Use Self-Play Script Directly
```bash
# Start self-play with flexible interruption
python scripts/run_large_selfplay.py \
  --model_path checkpoints/experiment/epoch4_mini1.pt.gz \
  --num_games 100000 \
  --output_dir data/sf25/selfplay_$(date +%Y%m%d_%H%M%S)

# Later, process whatever was generated
python scripts/training_pipeline.py \
  --model-path checkpoints/experiment/epoch4_mini1.pt.gz \
  --selfplay-dir data/sf25/selfplay_20250805_143022 \
  --no-selfplay
```

#### Option 2: Use Pipeline with Existing Data
```bash
# Process existing self-play data through to training
python scripts/training_pipeline.py \
  --model-path checkpoints/experiment/epoch4_mini1.pt.gz \
  --selfplay-dir data/sf25/selfplay_20250805_143022
```

This approach maintains the flexibility of your original workflow while providing the benefits of the new pipeline structure.

### Configuration

```bash
# Recommended settings for your setup
--num-workers 3        # 2-5 workers as you mentioned
--num-games 100000     # Total games across all workers
--batch-size 128       # Optimized for your GPU
--cache-size 60000     # Memory-efficient caching
```

### Performance Considerations

- **GPU utilization**: Each worker uses the same GPU, so batch size matters more than worker count
- **CPU utilization**: Multiple workers can utilize multiple CPU cores for game generation
- **Memory**: Each worker maintains its own cache and batch processing
- **I/O**: Streaming save prevents data loss during long runs

## Error Handling and Recovery

### Graceful Shutdown

The pipeline supports graceful shutdown via:
- `Ctrl+C` interruption
- Signal handling
- Progress saving and resume capability

### Resume Capability

- **Self-play**: Each worker saves progress independently
- **Shuffling**: Bucket-based approach allows resuming from any point
- **Training**: Checkpoint-based resume with hyperparameter validation

### Logging

Comprehensive logging includes:
- Step-by-step progress
- Performance metrics
- Error details and stack traces
- Configuration validation

Log files are saved to `logs/pipeline_YYYYMMDD_HHMMSS.log`

## Testing

### Validation Script

Run the test suite to validate the pipeline:

```bash
python scripts/test_pipeline.py
```

This tests:
- Import functionality
- Configuration creation
- Pipeline instantiation
- Argument parsing

### Dry Run

To test configuration without running:

```bash
python scripts/training_pipeline.py \
  --model-path checkpoints/experiment/epoch4_mini1.pt.gz \
  --num-games 100 \
  --no-training
```

## Migration from Bash Script

### Key Improvements

1. **Direct function calls** instead of subprocess
2. **Better error handling** with proper exception propagation
3. **Progress tracking** with resume capability
4. **Modular design** for easier testing and maintenance
5. **Configuration validation** before execution
6. **Automatic cleanup** of intermediate files

### Equivalent Commands

| Bash Script | New Pipeline |
|-------------|--------------|
| `run_large_selfplay.py` | `--run-selfplay` (default) |
| `preprocess_selfplay_data.py` | `--run-preprocessing` (default) |
| `process_all_trmph.py` | `--run-trmph-processing` (default) |
| `shuffle_processed_data.py` | `--run-shuffling` (default) |
| `hyperparam_sweep.py` | `--run-training` (default) |

## Best Practices

### Performance Optimization

1. **Worker count**: Use 2-5 workers for self-play
2. **Batch size**: Optimize for your GPU memory
3. **Cache size**: Balance memory usage and performance
4. **Chunk size**: Larger chunks reduce I/O overhead

### Data Management

1. **Cleanup**: Enable automatic cleanup to save disk space
2. **Validation**: Use existing data validation tools
3. **Backup**: Keep important intermediate results
4. **Monitoring**: Check log files for issues

### Configuration

1. **Version control**: Track configuration changes
2. **Experimentation**: Use different run timestamps for experiments
3. **Reproducibility**: Set random seeds for consistent results
4. **Documentation**: Document successful configurations

## Troubleshooting

### Common Issues

1. **Model not found**: Check model path and epoch/mini numbers
2. **Memory errors**: Reduce batch size or cache size
3. **Worker failures**: Check GPU memory and system resources
4. **Data corruption**: Validate input data and check logs

### Debug Mode

For debugging, use minimal settings:

```bash
python scripts/training_pipeline.py \
  --model-path checkpoints/experiment/epoch4_mini1.pt.gz \
  --num-games 100 \
  --num-workers 1 \
  --chunk-size 1000 \
  --max-workers-trmph 1
```

### Log Analysis

Check log files for:
- Step completion status
- Performance metrics
- Error messages and stack traces
- Configuration validation results 