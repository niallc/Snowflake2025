# Snowflake2025 - Hex AI Training Framework

A modern PyTorch implementation of a Hex AI training system, revamping the 2018 'Snowflake' project to create a stronger Hex AI using current best practices.

**Repository**: https://github.com/niallc/Snowflake2025

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/niallc/Snowflake2025.git
cd Snowflake2025
```

### 2. Set Up Environment
```bash
# Create and activate virtual environment
python -m venv hex_ai_env
source hex_ai_env/bin/activate  # On Windows: hex_ai_env\Scripts\activate

# Install dependencies (includes PyTorch)
pip install -r requirements.txt

# Check available devices (optional but recommended)
python scripts/check_device.py

# Optional: Optimize PyTorch installation for your hardware
# The above installation should work for most cases, but you can optimize:

# For CUDA GPUs (NVIDIA):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only systems:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For Apple Silicon (MPS) - reinstall from default PyPI for best compatibility:
# pip install torch torchvision torchaudio

# Set up environment
export PYTHONPATH=.
```

### 3. Create Required Directories
```bash
# Create all required directories (first-time setup only)
python scripts/setup_directories.py
```

### 4. Get Model and Data Files
**Important**: The repository does not include trained models or training data.

- **For playing**: Contact Niall Cardin (niallc@gmail.com) to get the latest model file
- **For training**: Get training data from Niall or generate using self-play with an existing model
- Place model files in the `checkpoints/` directory
- Update `hex_ai/inference/model_config.py` with the correct model path

## Quick Start

### For Coding Agents
If you're a coding agent working on this project:

1. **Quick Setup Check**: `python scripts/agent_setup.py`
2. **Environment Validation**: `python scripts/validate_environment.py`
3. **Common Commands**: `make help`
4. **Detailed Guidance**: See `AGENT_GUIDANCE.md`

**⚠️ IMPORTANT**: This project requires:
- Virtual environment: `hex_ai_env`
- PYTHONPATH: `export PYTHONPATH=.`
- Never skip environment checks in code!

## Main Entry Points

### Web Play
Play against trained models in your browser:
```bash
source hex_ai_env/bin/activate
PYTHONPATH=. python -m hex_ai.web.app --port 5001
```
Then open http://localhost:5001 in your browser.

### Training
The main training pipeline handles data collection, preprocessing, and model training:
```bash
source hex_ai_env/bin/activate
PYTHONPATH=. python scripts/training_pipeline.py \
  --use-current-best-model \
  --no-selfplay \
  --raw-trmph-data-dirs data/sf25/oct6 data/sf25/oct7 \
  --run-game-collection \
  --override-checkpoint-hyperparameters \
  --learning-rate 4.8e-3 \
  --processed-data-dirs data/processed/sf18_shuffled \
  --shard-ranges "204-207" "0097-0098" \
  --validation-dirs data/processed/sf18_shuffled \
  --validation-shard-ranges "498-498" "98-99"
```

For already processed data, use:
```bash
PYTHONPATH=. python scripts/training_pipeline.py \
  --use-current-best-model \
  --no-preprocessing --no-trmph-processing --no-shuffling \
  --processed-data-dirs data/processed/sf18_shuffled
```

### Model Discovery
Find the most recent trained models in `hex_ai/inference/model_config.py`. This is the central configuration for model paths used throughout the project.

## Project Structure

```
Snowflake2025/
├── hex_ai/                    # Core library modules
│   ├── web/                   # Web interface for playing
│   ├── inference/             # Model loading and inference
│   └── training/              # Training utilities
├── scripts/                   # Main entry point scripts
│   ├── training_pipeline.py   # Main training script
│   └── hyperparam_sweep.py    # Hyperparameter tuning
├── data/                      # Game data and processed shards
├── checkpoints/               # Training checkpoints
└── requirements.txt           # Dependencies
```

## Key Features

- **Web Interface**: Play against trained models in your browser
- **Comprehensive Training**: Full pipeline from data collection to model training
- **Model Management**: Centralized model configuration and discovery
- **Memory-Safe**: Built-in memory monitoring and emergency shutdown
- **Experiment Tracking**: CSV logging of all training metrics
- **Smart Checkpointing**: Strategic checkpoint retention to save space
- **GPU Acceleration**: Full CUDA support for faster training
- **Cross-Platform**: Works on macOS and Windows

## Running Tests

To run tests that import from the `hex_ai` package, you must set the `PYTHONPATH` to the project root. This ensures that imports like `from hex_ai...` work correctly.

```bash
source hex_ai_env/bin/activate
PYTHONPATH=. pytest tests/
```

Or for a specific test file:

```bash
PYTHONPATH=. pytest tests/test_streaming_augmented_processed_dataset.py
```

Run these commands from the project root directory.

## Troubleshooting

### Common Issues

**Missing directories error**: If you get errors about missing directories like `data/tournament_play` or `checkpoints/`, run:
```bash
python scripts/setup_directories.py
```

**Model not found**: If the web interface can't find models:
1. Ensure you have a model file in the `checkpoints/` directory
2. Check that `hex_ai/inference/model_config.py` points to the correct model path
3. Contact Niall Cardin (niallc@gmail.com) for the latest model file

**Training data not found**: If training fails due to missing data:
1. Get training data from Niall Cardin (niallc@gmail.com)
2. Or generate data using self-play with an existing model
3. Ensure data is in the correct format in `data/processed/` directories

**Environment issues**: If you get import errors:
```bash
# Ensure virtual environment is activated
source hex_ai_env/bin/activate

# Ensure PYTHONPATH is set
export PYTHONPATH=.

# Validate environment
python scripts/validate_environment.py
```

**PyTorch not found**: If you get "no module named torch":
```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA GPU support:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For Apple Silicon (MPS) - default installation is optimal:
# pip install torch torchvision torchaudio
```

**Device detection**: Check what devices are available:
```bash
python scripts/check_device.py
```

**Performance optimization**: The code automatically detects and uses the best available device:
- **CUDA**: Best for NVIDIA GPUs
- **MPS**: Good for Apple Silicon Macs  
- **CPU**: Fallback for all systems
