# Hex AI Network & Training Methodology Update Plan

## Goals
- **Produce a stronger, more robust Hex AI model** by modernizing the network architecture and training pipeline.
- **Adopt best practices from state-of-the-art game AIs** (AlphaZero, KataGo, ELF, etc.).
- **Prioritize high-impact, low-complexity changes** to maximize gains with minimal disruption.
- **Ensure all changes are well-tested, maintainable, and consistent with project style.**

## Summary Table: Proposed Changes
| Change                                      | Impact      | Certainty | Complexity | Notes/Discussion |
|----------------------------------------------|-------------|-----------|------------|------------------|
| **Add player-to-move input channel**         | High        | High      | Low        | Essential for value head; standard in modern AIs |
| **Wider initial convolution (e.g., 5x5)**    | Medium-High | High      | Low        | Captures more local context; minor compute cost |
| **Remove pooling layers**                    | Medium      | High      | Low        | Preserves board resolution; standard for board games |
| **Remove or minimize dropout**               | Low         | Medium    | Low        | Dropout rarely used in board game CNNs; discuss if any regularization is needed |
| **Zero-initialize value head bias**          | Medium      | High      | Low        | Helps value head start neutral; trivial to implement |
| **Gradient clipping**                        | Medium      | High      | Low        | Improves stability; easy to add |
| **Mixed precision training (AMP)**           | Medium      | High      | Medium     | Faster training, less memory; requires some code changes and testing |
| **Learning rate schedule (higher initial LR)**| Medium      | High      | Low        | Try higher LR (e.g., 0.01); scheduler handles decay |
| **SE blocks (Squeeze-and-Excitation)**       | Low-Med     | Low       | Medium-High| Optional; adds complexity; skip for now unless strong reason |
| **Output order (row-major)**                 | Essential   | High      | Already done | Confirmed consistent; no change needed |

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
- **How:** Remove or replace pooling layers with stride-1 convolutions.
- **Certainty:** High. **Impact:** Medium. **Complexity:** Low.

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

## Medium-Impact or Medium-Complexity Changes
### 7. Mixed Precision Training (AMP)
- **Motivation:** Faster training, less memory usage on modern GPUs.
- **How:** Use `torch.cuda.amp.autocast()` and `GradScaler` in training loop.
- **Certainty:** High. **Impact:** Medium. **Complexity:** Medium (requires careful testing).
- **Discussion:** Should be tested on your hardware; may require debugging for edge cases.

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

## Confirmed/No-Change Items
### 10. Output Order (Row-Major)
- **Motivation:** Consistency with data pipeline and conversion utilities.
- **How:** Confirmed already correct; no change needed.
- **Certainty:** High. **Impact:** Essential. **Complexity:** None.

## Discussion Points / Unresolved Questions
- **Dropout:** Should we remove it entirely, or keep a very small value for regularization? (Low impact, low complexity)
- **Mixed Precision:** Is your hardware (GPU/driver/PyTorch version) fully compatible? (Medium complexity, medium impact)
- **SE Blocks:** Worth considering only after all other changes are stable and tested.

## Implementation Tips
- **Zero-Initializing Value Head Bias:**
  - In PyTorch, after creating the value head's final linear layer, set `layer.bias.data.zero_()`.
- **Gradient Clipping:**
  - In your training loop, after `loss.backward()`, call `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` before `optimizer.step()`.
- **Mixed Precision:**
  - Use `with torch.cuda.amp.autocast():` for forward and loss computation, and `torch.cuda.amp.GradScaler()` for scaling gradients.
- **Wider Initial Conv:**
  - Change the kernel size and padding in the first conv layer. E.g., `nn.Conv2d(3, 64, kernel_size=5, padding=2)`.
- **Removing Pooling:**
  - Replace pooling layers with stride-1 convolutions or simply remove them to preserve spatial dimensions.

## Staged Implementation Plan
1. **Implement high-impact, low-complexity changes:** Input channel, wider conv, remove pooling, zero-init value head, gradient clipping, LR schedule.
2. **Test and validate:** Update tests, run small training runs, confirm improvements.
3. **Add medium-complexity improvements:** Mixed precision, if hardware supports.
4. **Review and discuss optional/low-impact changes:** Dropout, SE blocks.
5. **Clean up:** Remove obsolete code, update documentation, ensure consistency.

## Additional Notes
- **No Need to Mask Player Channel:** The player-to-move channel does not need to be masked for legal moves; legality is already encoded in the blue/red channels.
- **Future-Proofing:** This approach aligns with best practices in the field and will make future model improvements and comparisons easier. 