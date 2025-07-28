# Snowflake2025 - Hex AI Training Framework

A modern PyTorch implementation of a Hex AI training system, revamping the 2018 'Snowflake' project to create a stronger Hex AI using current best practices.

## Current Status

- ✅ **Data Processing**: Complete pipeline for processing .trmph game files into sharded pickle format
- ✅ **Model Architecture**: Two-headed ResNet with policy and value heads
- ✅ **Training Framework**: Full training loop with CSV logging and checkpoint management
- ✅ **Hyperparameter Tuning**: Automated tuning system with experiment tracking
- ✅ **Windows Support**: Verified setup for Windows with GPU acceleration

## Quick Start

### Setup
```bash
# Create environment
conda create -n hex_ai python=3.9
conda activate hex_ai

# Install PyTorch with CUDA (Windows)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Training
```bash
# Run hyperparameter tuning
python hyperparameter_tuning.py

# Monitor progress
python monitor_progress.py

# Train with specific configuration
python train_real_data.py

# Validate checkpoints
python scripts/validate_checkpoints.py --audit-all
```

### Data Processing
```bash
# Process raw .trmph files into shards
python resumable_shard_processing.py
```

## Project Structure

```
Snowflake2025/
├── hex_ai/                    # Core library modules
├── data/                      # Game data and processed shards
├── checkpoints/               # Training checkpoints
├── hyperparameter_results/    # Experiment results
├── requirements.txt           # Dependencies
├── hyperparameter_tuning.py  # Main tuning script
├── monitor_progress.py       # Progress monitoring
├── resumable_shard_processing.py  # Data processing
└── WINDOWS_SETUP.md          # Windows setup guide
```

## Key Features

- **Resumable Processing**: Can resume data processing from existing shards
- **Memory-Safe**: Built-in memory monitoring and emergency shutdown
- **Experiment Tracking**: CSV logging of all training metrics
- **Smart Checkpointing**: Strategic checkpoint retention to save space
- **Checkpoint Validation**: Tools to validate and audit checkpoint files
- **Checkpoint Compression**: Automatic gzip compression with 27.7% average space savings
- **GPU Acceleration**: Full CUDA support for faster training
- **Cross-Platform**: Works on macOS and Windows

## Documentation

- `WINDOWS_SETUP.md` - Complete Windows setup guide
- `hex_ai/data_formats.md` - Data format specifications
- `docs/checkpoint_format_specification.md` - Checkpoint format and validation

## Running Tests

To run tests that import from the `hex_ai` package, you must set the `PYTHONPATH` to the project root. This ensures that imports like `from hex_ai...` work correctly.

Example:

    PYTHONPATH=. pytest tests/

Or for a specific test file:

    PYTHONPATH=. pytest tests/test_streaming_augmented_processed_dataset.py

Run these commands from the project root directory.
