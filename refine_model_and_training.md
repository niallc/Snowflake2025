# Hex AI Network & Training Methodology Update Plan

## Goals
- **Produce a stronger, more robust Hex AI model** by modernizing the network architecture and training pipeline.
- **Adopt best practices from state-of-the-art game AIs** (AlphaZero, KataGo, ELF, etc.).
- **Prioritize high-impact, low-complexity changes** to maximize gains with minimal disruption.
- **Ensure all changes are well-tested, maintainable, and consistent with project style.**

---

## Current State (as of May 2024)
- **Model definition:** `hex_ai/models.py` contains a two-headed ResNet (policy/value heads). Input is currently `(batch_size, 2, 13, 13)` (no player-to-move channel yet).
- **Training flow:** Training is managed via `hex_ai/training.py` (Trainer class), with entry points in `hyperparameter_tuning_gpu_large.py` and `training_utils.py`.
- **Mixed precision:** Code for mixed precision exists, but MPS (Apple GPU) detection and usage may not be consistent or centralized. CUDA is checked, but MPS support may be missed, so mixed precision may not be used on Mac by default.
- **Pooling/downsampling:**
    - No explicit pooling layers in the ResNet body; downsampling is done via stride-2 convolutions (standard in ResNet).
    - There **is** a global average pooling layer (`nn.AdaptiveAvgPool2d((1, 1))`) at the end of the ResNet body, which is standard for classification/value heads.
    - **Note:** Some advice suggests using only stride-1 convolutions for board games to preserve spatial information. This is up for discussion (see below).
- **Dropout:** Present in the model, but the plan suggests minimizing or removing it.
- **No SE blocks** (good; optional/complex).
- **Output order:** Already row-major; no change needed.
- **Data pipeline:** Will need to update to add player-to-move channel.
- **No major code duplication in main model/training, but some legacy/utility code may be redundant.**

---

## Summary Table: Proposed Changes
| Change                                      | Impact      | Certainty | Complexity | Notes/Discussion |
|----------------------------------------------|-------------|-----------|------------|------------------|
| **Add player-to-move input channel**         | High        | High      | Low        | Essential for value head; standard in modern AIs |
| **Wider initial convolution (e.g., 5x5)**    | Medium-High | High      | Low        | Captures more local context; minor compute cost |
| **Remove pooling layers**                    | Medium      | High      | Low        | No explicit pooling, but global avg pooling is present; see notes |
| **Remove or minimize dropout**               | Low         | Medium    | Low        | Dropout rarely used in board game CNNs; discuss if any regularization is needed |
| **Zero-initialize value head bias**          | Medium      | High      | Low        | Helps value head start neutral; trivial to implement |
| **Gradient clipping**                        | Medium      | High      | Low        | Improves stability; easy to add |
| **Mixed precision training (AMP)**           | Medium      | High      | Medium     | Code exists, but MPS/Apple GPU support may not be active; see TODOs |
| **Learning rate schedule (higher initial LR)**| Medium      | High      | Low        | Try higher LR (e.g., 0.01); scheduler handles decay |
| **SE blocks (Squeeze-and-Excitation)**       | Low-Med     | Low       | Medium-High| Optional; adds complexity; skip for now unless strong reason |
| **Output order (row-major)**                 | Essential   | High      | Already done | Confirmed consistent; no change needed |
| **Centralize device/mixed precision detection** | Medium   | High      | Low        | Ensure all code uses a single utility for device selection and mixed precision (see TODOs) |
| **Audit and update all code to use device utility** | Medium | High | Low | Ensure consistency and correct use of MPS/Apple GPU |

---

## High-Impact, Low-Complexity Changes (Do First)
### 1. Add Player-to-Move Input Channel
- **Motivation:** Essential for value head to distinguish whose turn it is; standard in all strong game AIs.
- **How:** Add a third channel to the input tensor: all ones if BLUE's turn, all zeros if RED's (use `BLUE_PLAYER = 0`, `RED_PLAYER = 1`).
- **Certainty:** High. **Impact:** High. **Complexity:** Low.

### 2. Wider Initial Convolution
- **Motivation:** Captures more local context in the first layer; used in many strong AIs.
- **How:** Change first conv layer to 5x5 (or similar) kernel, adjust padding.
- **Certainty:** High. **Impact:** Medium-High. **Complexity:** Low.

### 3. Remove Pooling Layers
- **Motivation:** Pooling reduces spatial resolution, which is undesirable for board games.
- **How:** Remove or replace pooling layers with stride-1 convolutions. **Note:** The current model uses stride-2 convolutions for downsampling (standard in ResNet), and a global average pooling at the end. Some advice suggests using only stride-1 convolutions and removing global average pooling for board games. **This is up for discussion:** (default plan described at end)
    - **Pros of stride-1 only:** Maximum spatial fidelity, may help for small boards.
    - **Cons:** Very high memory/compute, may not be necessary for 13x13 boards.
    - **Global average pooling:** Standard for value/policy heads, but could be replaced with flattening or other approaches if full spatial info is desired.

### 4. Zero-Initialize Value Head Bias
- **Motivation:** Ensures value head starts with neutral predictions; standard trick.
- **How:** Set bias of final value head layer to zero after model creation.
- **Certainty:** High. **Impact:** Medium. **Complexity:** Low.

### 5. Gradient Clipping
- **Motivation:** Prevents exploding gradients, improves stability.
- **How:** Use `torch.nn.utils.clip_grad_norm_` in training loop.
- **Certainty:** High. **Impact:** Medium. **Complexity:** Low.

### 6. Learning Rate Schedule (Higher Initial LR)
- **Motivation:** Faster initial learning; scheduler handles decay.
- **How:** Try LR=0.01, use ReduceLROnPlateau or cosine annealing.
- **Certainty:** High. **Impact:** Medium. **Complexity:** Low.

---

## Medium-Impact or Medium-Complexity Changes
### 7. Mixed Precision Training (AMP)
- **Motivation:** Faster training, less memory usage on modern GPUs.
- **How:** Use `torch.cuda.amp.autocast()` and `GradScaler` in training loop. **Ensure MPS/Apple GPU is detected and used where available.**
- **Certainty:** High. **Impact:** Medium. **Complexity:** Medium (requires careful testing).
- **Discussion:** Should be tested on your hardware; may require debugging for edge cases.

---

## Low-Impact or Optional Changes
### 8. Remove or Minimize Dropout
- **Motivation:** Dropout is rarely used in board game CNNs; BatchNorm and data augmentation provide sufficient regularization.
- **How:** Remove dropout layers or set to a very low value (e.g., 0.05).
- **Certainty:** Medium. **Impact:** Low. **Complexity:** Low.
- **Discussion:** If you observe overfitting, consider keeping a small amount. Otherwise, safe to remove.

### 9. SE Blocks (Squeeze-and-Excitation)
- **Motivation:** Can improve performance, but adds complexity.
- **How:** Add SE blocks to residual blocks.
- **Certainty:** Low. **Impact:** Low-Med. **Complexity:** Medium-High.
- **Discussion:** Skip for now; revisit if you want to push for maximum strength later.

---

## Confirmed/No-Change Items
### 10. Output Order (Row-Major)
- **Motivation:** Consistency with data pipeline and conversion utilities.
- **How:** Confirmed already correct; no change needed.
- **Certainty:** High. **Impact:** Essential. **Complexity:** None.

---

## Additional TODOs (from code review)
1. **Centralize device/mixed precision detection:**
    - Create a utility function/module to detect and select device (CPU, CUDA, MPS) and whether mixed precision should be used.
    - Ensure all model, training, and inference code uses this utility for device selection and autocast context.
2. **Audit and update all code to use device utility:**
    - Refactor codebase to use the new utility everywhere device/mixed precision is relevant.
    - Test on M1 Mac to ensure MPS/mixed precision is actually used.

---

## Discussion Points / Unresolved Questions
- **Dropout:** Should we remove it entirely, or keep a very small value for regularization? (Low impact, low complexity)
- **Mixed Precision:** Is your hardware (GPU/driver/PyTorch version) fully compatible? (Medium complexity, medium impact)
- **SE Blocks:** Worth considering only after all other changes are stable and tested.
- **Stride-2 vs. Stride-1 Downsampling:** Should we use only stride-1 convolutions (no downsampling) for board games, or keep stride-2 as in standard ResNet? Should we keep or remove global average pooling? **Needs empirical testing and/or further research.**

---

## Implementation Tips
- **Zero-Initializing Value Head Bias:**
  - In PyTorch, after creating the value head's final linear layer, set `layer.bias.data.zero_()`.
- **Gradient Clipping:**
  - In your training loop, after `loss.backward()`, call `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` before `optimizer.step()`.
- **Mixed Precision:**
  - Use `with torch.cuda.amp.autocast():` for forward and loss computation, and `torch.cuda.amp.GradScaler()` for scaling gradients. For MPS, use `torch.autocast(device_type="mps")`.
- **Wider Initial Conv:**
  - Change the kernel size and padding in the first conv layer. E.g., `nn.Conv2d(3, 64, kernel_size=5, padding=2)`.
- **Removing Pooling:**
  - Replace pooling layers with stride-1 convolutions or simply remove them to preserve spatial dimensions. Consider also removing/replacing global average pooling if full spatial info is desired.

---

## Staged Implementation Plan
1. **Implement high-impact, low-complexity changes:** Input channel, wider conv, remove pooling/stride-2 if desired, zero-init value head, gradient clipping, LR schedule.
2. **Test and validate:** Update tests, run small training runs, confirm improvements.
3. **Add medium-complexity improvements:** Mixed precision, if hardware supports.
4. **Review and discuss optional/low-impact changes:** Dropout, SE blocks.
5. **Clean up:** Remove obsolete code, update documentation, ensure consistency.

---

## Additional Notes
- **No Need to Mask Player Channel:** The player-to-move channel does not need to be masked for legal moves; legality is already encoded in the blue/red channels.
- **Future-Proofing:** This approach aligns with best practices in the field and will make future model improvements and comparisons easier. 

---

## Stride-1 vs Stride-2 Downsampling: Compromise and Parameterization

- **Background:** In standard ResNet architectures, downsampling is performed using stride-2 convolutions, which quickly reduce spatial resolution. For board games like Hex, retaining spatial information is often more important than in natural image tasks.
- **Compromise:** Rather than removing all downsampling (which is very costly), we plan to make only the *first* downsampling step (layer2) use stride=1 (no downsampling), while keeping stride=2 for later layers. This keeps more spatial information in the early/mid network, at a moderate memory cost.
- **Rationale:**
    - This approach is used in some strong board game AIs (e.g., KataGo, ELF) to balance spatial fidelity and efficiency.
    - It allows the network to learn more local and mid-range patterns before compressing the feature map.
    - The memory/compute cost is much less than making all layers stride=1, but the benefit is significant for board games.
- **Parameterization:**
    - We will add a parameter to the model API (e.g., `first_downsample_stride` or `layer2_stride`) to allow easy experimentation with stride=1 vs stride=2 for the first downsampling layer.
    - This will be documented in the model's docstring and in this plan.
- **Implementation order:**
    - The first implementation step will be the addition of the player-to-move input channel (3rd channel), as this is the highest-impact change.
    - The stride parameterization will follow, with full documentation and clear API. 