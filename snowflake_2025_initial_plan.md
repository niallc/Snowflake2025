# Project Brief: Modernizing the Hex AI

## 1. Project Overview

The goal of this project is to create a modern, high-performance Hex AI by re-implementing an existing 2018 TensorFlow 1.1 project in PyTorch.

A possible new architecture will be a **two-headed ResNet** (ResNet-18 or similar) with a modern head design (Global Average Pooling). We might also save some time by reusing or adapting the project's proven data-processing logic in legacy_code while building the PyTorch model, dataset interface, and training loop from scratch to follow modern best practices.

**Note:** This document is a guide to get the project started. You are encouraged to overrule the guide and update it while creating a clean, efficient, and powerful program.

## 2. The Overall Plan

The development will proceed in five main phases:
1.  **Project Scaffolding:** Setting up a clean, organized repository.
2.  **The Data Pipeline:** Creating a PyTorch `Dataset` to interface with our pre-processed game data.
3.  **The Model Architecture:** Implementing the new model, e.g. two-headed ResNet.
4.  **The Training Loop:** Building a script to train the model on a GPU, with modern optimization and logging. We can start with an extensive library of existing games.
5.  **Validation:** Establishing a tournament to maesure each model iterations performance against previous rounds, to find the best models and identify rates of progress.

## 3. Detailed Next Steps (Coding Tasks)

### Task 3.1: The Data Pipeline (`hex_ai/dataset.py`)

Create a `HexDataset` class that interfaces with pre-processed, sharded NumPy data.

The implementation details can be figured out as we go, but an option might be:
* Inherit from `torch.utils.data.Dataset`.
* The `__init__` method should take a path to the processed data directory and create a list of all data shards.
* The `__getitem__` method should be responsible for loading a data point (board, policy target, value target) by index, converting the NumPy arrays to PyTorch tensors, and returning them.
* **Assumed Tensor Shapes (Update if necessary):**
    * `board_tensor`: `torch.float32`, shape `(2, 13, 13)`
    * `policy_tensor`: `torch.float32`, shape `(169)`
    * `value_tensor`: `torch.float32`, shape `(1)` (we'll use a single neuron output with BCE loss)

### Task 3.2: The Model Architecture (`hex_ai/models.py`)

Implement the new architecture. For example, a two-headed ResNet architecture following e.g.

1.  **`ResNetBlock` Module:** Create a `nn.Module` for a standard ResNet block. It should contain two sequences of `Conv2d` -> `BatchNorm2d` -> `ReLU`. It must handle residual connections, including projection shortcuts for when dimensions change.
2.  **`TwoHeadedResNet` Module:** Create the main model `nn.Module`.
    * **Body:** Start with an initial `Conv2d` layer, followed by a series of `ResNetBlock`s to form a ResNet-18. Use standard filter progressions (e.g., 64 -> 128 -> 256 -> 512), downsampling with a stride of 2 at the start of new filter stages.
    * **Head:**
        * After the ResNet body, use an `nn.AdaptiveAvgPool2d((1, 1))` layer.
        * **Policy Head:** An `nn.Linear` layer mapping the pooled features to `169` outputs.
        * **Value Head:** A separate `nn.Linear` layer mapping the pooled features to `1` output.
    * The `forward` method should return two values: `policy_logits`, `value_logit`.

### Task 3.3: The Training Script (`notebooks/train_model.ipynb`)

Set up the main training script to orchestrate the process.

* **Setup:** Instantiate the `HexDataset`, wrap it in a `DataLoader`, and instantiate the new  model. One possible setup for this could be the following, though suggestions for more modern, robust, mainainable, or efficient regimes are welcomed:
* **Loss Function:**
    * `policy_loss = nn.CrossEntropyLoss(policy_logits, policy_targets)`
    * `value_loss = nn.BCEWithLogitsLoss(value_logit, value_targets)`
    * `total_loss = policy_loss + value_loss`
* **Optimizer & Scheduler:**
    * Use `torch.optim.AdamW`.
    * Use `torch.optim.lr_scheduler.CosineAnnealingLR`.
* **Logging:** Integrate `wandb` to log the losses and learning rate for each step.

---
## 4. Suggested Workflow and Development Plan

This section provides a suggested approach to the development process.

### 4.1 The `legacy_code` Directory

The `./legacy_code` directory contains files from the original project, such as `ReadHexGamesFiles.py` and `FileConversion.py`. These files contain the battle-tested logic for:
* Parsing `.trmph` game files.
* Generating training positions from game histories.
* Performing data augmentation via rotation and reflection.

This code can be used to create a one-off pre-processing script to convert the raw game data into sharded NumPy arrays for our new `HexDataset` to consume. Or it can simply be used as a way of understanding the locations and structures of the original data, or as a starting point for new and more efficient processing code.

There is no requirement to reuse any of this legacy code. It is provided primarily as a reference. If it's simpler to rewrite a particular utility from scratch in a more modern or efficient way, please do so.

### 4.2 Development Workflow

We recommend the following workflow to maintain a clean and reproducible project:
1.  **Develop Locally:** All core library code in the `hex_ai/` directory should be developed and tested on a local machine.
2.  **Commit to GitHub:** Use Git for version control with frequent, small commits. Push changes regularly to a central GitHub repository to serve as the single source of truth.
3.  **Execute in Colab:** The `notebooks/train_model.ipynb` is the execution environment. To run an experiment, it should start by cloning the latest version of the repository from GitHub (`!git clone ...`). This ensures that training runs are always performed on the most up-to-date, version-controlled code. The purpose of running in colab is only to use modern GPUs to save time. Early runs will probably be run locally, until we're ready to use more resources for a full train.

### 4.3 Phased Development Schedule

One possible logical path for development would be to build the components in order of dependency:

* **Phase A: Data Interface (`dataset.py`):** Start here. Implement the `HexDataset` class. The immediate goal is to successfully load a single pre-processed data point and return it as a correctly shaped Torch tensor.
* **Phase B: Model Architecture (`models.py`):** With a way to get data, implement the `TwoHeadedResNet`. The goal is to ensure a batch of data can pass through the model's `forward()` method without shape errors.
* **Phase C: Core Training Step (`train_model.ipynb`):** Implement the core logic of the training loop. The goal is to run a single training step on one batch of data, calculating the combined loss and successfully executing `loss.backward()` and `optimizer.step()`.
* **Phase D: Full Training & Logging:** Expand the loop to run for a full epoch. Integrate `wandb` to log metrics. The goal is to ensure the training process is stable and that the loss decreases over time.
* **Phase E: Validation:** Once a model is trained, implement the tournament logic to validate its performance.

### 4.4 Testing and Documentation

* **Testing:** Please write unit tests for critical utility functions where possible (e.g., any new data transformation or game logic functions). For the main components, "integration tests" are key—for example, a test that confirms the `DataLoader` and `TwoHeadedResNet` can work together for one forward/backward pass is extremely valuable.
* **Documentation:** Use clear docstrings, focusing on the **"why"** for major architectural decisions in key modules (`models.py`, `dataset.py`). The goal is to help a future developer quickly understand the high-level design. Avoid low-level comments that merely state *what* the code is doing. Only log changes to a CHANGELOG file, other documentation should be kept clean and _not_ reference every change in design, only the main current design points at any given time.

---
## 5. Design Decisions and Discussion Log

### 5.1 Initial Discussion (Current Session)

**Data Availability:**
- Raw game data is available in `data/twoNetGames/` directory
- Contains ~100+ `.trmph` format game files (various sizes from 3KB to 9MB)
- Legacy code contains utilities to convert `.trmph` to other formats and create sharded data

**Hardware & Environment:**
- Development machine: M1 Mac (no dedicated GPU)
- Can handle test runs and slow training (~1M games/day with old architecture)
- Plan to use Google Colab or other GPU resources for faster training
- Environment: Python virtual environment with PyTorch, NumPy, Pandas, Matplotlib, WandB, Jupyter

**Architecture Decisions:**
- Starting with ResNet-18 (proven, efficient, good for 1-10M games)
- Open to attention-based architectures if resources allow
- Flexible on precise design - can iterate and improve
- Board representation: `(2, 13, 13)` tensors (2 channels for two players)
- Policy head: 169 outputs (13x13 board positions)
- Value head: 1 output (binary classification)
- Modern practices: Mixed precision training, gradient checkpointing, efficient data loading

**Development Approach:**
- Start with simple architecture, upgrade later if needed
- Set up network and training system first, then format data accordingly
- Focus on maintainable, high-performance code
- Use modern PyTorch practices and best practices

**Project Structure:**
```
hex_ai/
├── __init__.py
├── dataset.py          # HexDataset class
├── models.py           # TwoHeadedResNet architecture
├── utils.py            # Utility functions
└── config.py           # Configuration constants

notebooks/
└── train_model.ipynb   # Training script

tests/
└── test_dataset.py     # Unit tests

data/                   # For processed data
└── processed/

requirements.txt
```

**Current Progress:**
- ✅ Project scaffolding complete with clean, modular structure
- ✅ Virtual environment set up with all dependencies
- ✅ Comprehensive documentation of data formats and legacy code analysis
- ✅ Detailed placeholder interfaces for data pipeline with type hints
- ✅ Basic test suite in place and passing
- ✅ Configuration system with constants and settings

**Next Steps:**
1. ✅ Set up project scaffolding with placeholder files
2. ✅ Create virtual environment and install dependencies
3. ✅ Document data formats and processing pipeline (`hex_ai/data_formats.md`)
4. ✅ Create detailed placeholder interfaces for data pipeline (`hex_ai/data_utils.py`)
5. ✅ Analyze modern architecture choices and resource requirements
6. **Next: Implement ResNet-18 model (Phase B)** with modern best practices
7. **Then: Implement data pipeline (Phase A)** once model requirements are clear
8. **Then: Build training loop (Phase C)** with mixed precision and efficient loading
9. **Then: Add logging and full training (Phase D)**
10. **Then: Implement validation (Phase E)**
11. **Future: Experiment with attention models if resources allow**