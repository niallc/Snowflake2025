# Project Brief: Modernizing the Hex AI

## 1. Project Overview

The goal of this project is to create a modern, high-performance Hex AI by re-implementing an existing 2018 TensorFlow 1.1 project in PyTorch.

The new architecture will be a **two-headed ResNet** (ResNet-18 or similar) with a modern head design (Global Average Pooling). The core principle is to **reuse the project's proven data-processing logic** while building the PyTorch model, dataset interface, and training loop from scratch to follow modern best practices.

**Note:** This document is a strategic guide to get the project started. You are encouraged to ask clarifying questions and suggest improvements or alternative implementations that align with the goals of creating a clean, efficient, and powerful program.

## 2. The Overall Plan

The development will proceed in five main phases:
1.  **Project Scaffolding:** Setting up a clean, organized repository.
2.  **The Data Pipeline:** Creating a PyTorch `Dataset` to interface with our pre-processed game data.
3.  **The Model Architecture:** Implementing the new two-headed ResNet model.
4.  **The Training Loop:** Building a script to train the model on a GPU, with modern optimization and logging.
5.  **Validation:** Establishing a tournament to measure the new model's performance against the old one.

## 3. Detailed Next Steps (Coding Tasks)

### Task 3.1: The Data Pipeline (`hex_ai/dataset.py`)

Create a `HexDataset` class that interfaces with pre-processed, sharded NumPy data.

* Inherit from `torch.utils.data.Dataset`.
* The `__init__` method should take a path to the processed data directory and create a list of all data shards.
* The `__getitem__` method should load a data point (board, policy target, value target) by index, convert the NumPy arrays to PyTorch tensors, and return them.
* **Assumed Tensor Shapes (Update if necessary):**
    * `board_tensor`: `torch.float32`, shape `(2, 13, 13)`
    * `policy_tensor`: `torch.float32`, shape `(169)`
    * `value_tensor`: `torch.float32`, shape `(1)` (we'll use a single neuron output with BCE loss)

### Task 3.2: The Model Architecture (`hex_ai/models.py`)

Implement the two-headed ResNet architecture.

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

* **Setup:** Instantiate the `HexDataset`, wrap it in a `DataLoader`, and instantiate the `TwoHeadedResNet` model.
* **Loss Function:**
    * `policy_loss = nn.CrossEntropyLoss(policy_logits, policy_targets)`
    * `value_loss = nn.BCEWithLogitsLoss(value_logit, value_targets)`
    * `total_loss = policy_loss + value_loss`
* **Optimizer & Scheduler:**
    * Use `torch.optim.AdamW`.
    * Use `torch.optim.lr_scheduler.CosineAnnealingLR`.
* **Training Loop:** Write a standard training loop that iterates through epochs and batches, calculates the combined loss, performs backpropagation, and updates the weights.
* **Logging:** Integrate `wandb` to log the losses and learning rate for each step.