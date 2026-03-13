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

# Optional: Check what devices are available
python scripts/check_device.py

# Install this repo as an editable package (recommended)
# This makes `import hex_ai` work without PYTHONPATH hacks.
pip install -e .
```

> **Windows users**: For a detailed walkthrough that covers installing prerequisites (Git, Python, Build Tools), creating a
> PowerShell virtual environment, installing the correct PyTorch wheel, and checking out the `oct10` branch, see
> [`docs/windows_setup.md`](docs/windows_setup.md).

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
If you're a coding agent working on this project, start with:

1. `AGENTS.md` for repo-specific working rules
2. `write_ups/Current_Network_vs_KataGo_Gumbel_2026-03-13.md` for the active network-design review
3. `write_ups/Training_Architecture_Change_Discussion_2026-03-13.md` for the implementation-facing architecture roadmap
4. `docs/value_head_specification.md` for the current value-head/runtime contract

Important environment note:
- activate `hex_ai_env`
- use `pip install -e .`
- this repo enforces virtualenv usage via `hex_ai/__init__.py`

Board-size note:
- the current default training/inference configuration is `13x13`
- the project direction is toward board-size-parameterized play/training, so avoid introducing new docs/comments that imply `13x13` is a permanent design limit

## Main Entry Points

### Web Play
Play against trained models in your browser:
```bash
source hex_ai_env/bin/activate
python -m hex_ai.web.app --port 5001
```
Then open http://localhost:5001 in your browser.

### Training
The main training pipeline handles data collection, preprocessing, and model training:
```bash
source hex_ai_env/bin/activate
python scripts/training_pipeline.py \
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
python scripts/training_pipeline.py \
  --use-current-best-model \
  --no-preprocessing --no-trmph-processing --no-shuffling \
  --processed-data-dirs data/processed/sf18_shuffled
```

### Model Discovery
Find the most recent trained models in `hex_ai/inference/model_config.py`. This is the central configuration for model paths used throughout the project.

### Architecture Notes
For current neural-net design work, these are the main references in the repo:

- `write_ups/Current_Network_vs_KataGo_Gumbel_2026-03-13.md`
- `write_ups/Training_Architecture_Change_Discussion_2026-03-13.md`
- `docs/value_head_specification.md`
- `hex_ai/models.py`

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

If you installed the repo in editable mode (`pip install -e .`), you can run tests without setting `PYTHONPATH`.

```bash
source hex_ai_env/bin/activate
pytest tests/
```

Or for a specific test file:

```bash
pytest tests/test_streaming_augmented_processed_dataset.py
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

# Ensure editable install is present (one-time per venv)
pip install -e .

# Quick import check
python -c "import hex_ai; print('hex_ai import OK')"
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
