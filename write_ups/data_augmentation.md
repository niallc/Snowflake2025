# Data Augmentation for Hex AI

This document describes the goals, technical details, and caveats of data augmentation for Hex AI training data.

## 1. Goals of Data Augmentation

- **Increase effective dataset size** by generating new, logically equivalent board positions from each original position.
- **Improve generalization** by exposing the model to all board symmetries, reducing overfitting to specific board layouts or move orders.
- **Ensure label consistency**: All augmented boards must have correct policy, value, and player-to-move labels.

## 2. Board Manipulations (Symmetries)

For a 13x13 Hex board, there are 4 main symmetries we use for augmentation:

### a. Original
- The board as played.

### b. 180° Rotation (no color swap)
- Rotate the board 180°.
- **No color swap**: Blue stays blue, red stays red.
- Board edges retain their meaning (top/bottom = blue, left/right = red).
- The player-to-move does **not** change.

### c. Long Diagonal Reflection + Color Swap
- Reflect the board along the long diagonal (top-left to bottom-right).
- **Swap colors**: Blue <-> Red.
- Board edges swap meaning (top/bottom <-> left/right).
- The player-to-move **must be swapped** (blue <-> red).
- The value label (winner) **must be swapped** (blue win <-> red win).
- The policy label must be transformed accordingly.

### d. Short Diagonal Reflection + Color Swap
- Reflect the board along the short diagonal (top-right to bottom-left).
- **Swap colors**: Blue <-> Red.
- Board edges swap meaning (top/bottom <-> right/left, but mirrored).
- The player-to-move **must be swapped** (blue <-> red).
- The value label (winner) **must be swapped** (blue win <-> red win).
- The policy label must be transformed accordingly.

#### Note
- **Diagonal reflections** always require color swap to preserve logical equivalence under the swap rule.
- **180° rotation** does **not** require color swap.

## 3. Label Adjustments

For each augmentation, the following must be handled:

- **Policy label**: Must be transformed using the same symmetry as the board.
- **Value label**: For color-swapping symmetries, the winner must be swapped (1.0 <-> 0.0).
- **Player-to-move channel**: For color-swapping symmetries, the player-to-move must be swapped (blue <-> red).

## 4. Player-to-Move and Color Swap

- The player-to-move channel is used as the 3rd channel in the model input.
- When a symmetry swaps colors (i.e., diagonal reflections), the player-to-move must also be swapped:
    - If it was blue's turn, after color swap it is red's turn, and vice versa.
- This is essential for label consistency and correct training.

## 5. TODO / Open Questions

- **Short diagonal reflection**: The current implementation may not be correct. It should:
    - Leave pieces on the short diagonal unchanged.
    - Move other pieces to their mirror position across the short diagonal.
    - Swap colors.
- **Policy transformation**: Must match the board transformation exactly.
- **Testing**: Visualize all augmentations and labels to verify correctness.

---

*This document is a living reference. Please update as the augmentation logic is refined or new symmetries/labels are added.* 