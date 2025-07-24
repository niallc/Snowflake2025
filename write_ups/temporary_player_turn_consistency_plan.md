# Temporary Plan: Ensuring Player Turn Consistency in Hex AI

## Background

In Hex AI, the neural network input includes a player-to-move channel (the 3rd channel). This channel is critical for correct value and policy predictions, especially in positions where the next move determines the winner. Inconsistencies in how this channel is set between training and inference can lead to systematic errors, such as the model predicting the wrong winner in "critical move" positions.

## Symptom

- The CLI (`scripts/simple_inference_cli.py`) produces value head predictions that appear to have the player turn flipped compared to what is expected from the training process.
- Example: In positions where Blue should win by playing at g7, the model predicts a high probability for Red, and vice versa.
- This suggests a mismatch in how the player-to-move channel is set between training data creation and inference.

## Broader Task

- Ensure that the logic for determining the player-to-move is **identical** and **centralized** for both training and inference.
- Make the player-to-move information explicit in processed data, rather than inferring it from the board tensor (which can be ambiguous after augmentation or color swaps).

## Step-by-Step Plan

### 1. Update Data Processing Pipeline
- **File:** `hex_ai/data_processing.py`
  - **Class:** `DataProcessor`
  - **Method:** `_convert_games_to_tensors`
  - **Action:**
    - Add a new dictionary element to each example for the player turn (e.g., `example['player_to_move']`).
    - Use a central utility (e.g., `get_player_to_move_from_board` or a new function) to assign this value directly from the TRMPH string/move list.
    - Ensure this is done before any augmentation or color swapping.

### 2. Update Data Loading Code
- **Files:**
  - `hex_ai/data_pipeline.py` (e.g., `StreamingAugmentedProcessedDataset`)
  - Any other code that reads processed data
- **Action:**
  - Update code to read and use the explicit `player_to_move` field from the processed data, rather than inferring it from the board tensor.
  - Pass this value through to the model input as the 3rd channel.

### 3. Add Tests for Player Turn Utilities
- **Files:**
  - `tests/test_value_utils.py` (or a new test file)
- **Action:**
  - Add tests that check the player-to-move assignment for a series of positions, both from TRMPH strings and from processed data.
  - Include edge cases (e.g., equal number of blue/red pieces, after augmentation, etc.).

### 4. Centralize and Standardize Utilities
- **Files:**
  - `hex_ai/value_utils.py` (or another central location)
- **Action:**
  - Ensure all code that processes a TRMPH format game into the network input uses the same utility for player-to-move assignment.
  - Refactor as needed to remove duplicate or inconsistent logic.

## Relevant Files, Classes, and Functions

- `hex_ai/data_processing.py` — `DataProcessor._convert_games_to_tensors`
- `hex_ai/data_utils.py` — `get_player_to_move_from_board`, `create_board_from_moves`, `preprocess_example_for_model`
- `hex_ai/data_pipeline.py` — `StreamingAugmentedProcessedDataset`, `get_augmented_tensor_for_index`, `_transform_example`
- `hex_ai/inference/simple_model_inference.py` — `_create_board_with_correct_player_channel`
- `hex_ai/value_utils.py` — (potential new/centralized utilities)
- `tests/test_value_utils.py` — (add new tests)

## Next Steps

- [x] Update data processing to store player-to-move explicitly
- [x] Ensure player-to-move field is retained through data shuffling (scripts/shuffle_processed_data.py)
- [ ] Update data loading to use explicit player-to-move
- [ ] Add comprehensive tests for player-to-move logic
- [ ] Refactor all code to use centralized utilities for player-to-move

---

## Coding practices
- Better to go slowly and carefully and get things right.
- Aim for simple small functions, avoiding complex logic.
- Avoid code duplication, checking for existing utilities.
 - If partial duplication is hard to avoid and potentially dead code paths are generated, note the concern in write_ups/duplicate_code.md
- Once code is write -- add and run the tests.
 - If there are failures with tests and you know how to fix them, just fix them and rerun. No need to check with me.
 - To run code, including tests, you'll need to use PYTHONPATH=. to get over import issues.

---

## Agent Behaviour.
- If you know what to do, keep going.
- If you're not sure how to proceed, e.g. you come across an unexpected problem that isn't just a mistake in a test, and you've already written code, stop and ask.
- Prefer to raise an exception if something unexpected happens, like a missing field in the data. In general I'm very keen to avoid silent failures, so I want to avoid fallbacks in general.

---

*This document is temporary and will be updated as the plan is executed and new issues are discovered.* 