# New Files Inventory - Code Reorganization

## Overview
This document lists all new files created during the code reorganization and their specific purposes.

## New Files Created

### 1. `hex_ai/trmph_processing/` - Core Processing Module

#### `hex_ai/trmph_processing/__init__.py`
- **Purpose**: Module initialization and exports
- **Contents**: Exports all public classes and functions from the module
- **Exports**: `ProcessingConfig`, `TRMPHProcessor`, `ParallelProcessor`, `SequentialProcessor`, `process_single_file_worker`, `process_single_file_direct`, `main`

#### `hex_ai/trmph_processing/config.py`
- **Purpose**: Configuration management for TRMPH processing
- **Key Class**: `ProcessingConfig`
- **Responsibilities**:
  - Configuration validation
  - Serialization for multiprocessing
  - Default value management
  - Path handling

#### `hex_ai/trmph_processing/processor.py`
- **Purpose**: Processing orchestrators and multiprocessing logic
- **Key Classes**:
  - `ParallelProcessor`: Handles parallel processing using ProcessPoolExecutor
  - `SequentialProcessor`: Handles sequential processing for debugging
  - `TRMPHProcessor`: Main orchestrator that coordinates file discovery and processing
- **Responsibilities**:
  - File discovery and filtering
  - Worker coordination
  - Result collection and logging
  - Error handling

#### `hex_ai/trmph_processing/workers.py`
- **Purpose**: Worker functions for multiprocessing
- **Key Functions**:
  - `process_single_file_worker`: Main worker function for multiprocessing
  - `process_single_file_direct`: Direct file processing without BatchProcessor state
  - `validate_examples_data`: Data validation for training examples
- **Responsibilities**:
  - Individual file processing
  - Data validation
  - Error isolation
  - File output with atomic writes

#### `hex_ai/trmph_processing/cli.py`
- **Purpose**: Command-line interface logic
- **Key Functions**:
  - `main()`: Main CLI entry point
  - `parse_arguments()`: Argument parsing
  - `setup_logging()`: Logging configuration
  - `create_config_from_args()`: Config creation from CLI args
  - `process_files()`: File processing coordination
  - `create_combined_dataset()`: Combined dataset creation
  - `print_output_summary()`: Results summary
- **Responsibilities**:
  - CLI argument parsing
  - Logging setup
  - Configuration creation
  - Processing coordination
  - Output reporting

### 2. `hex_ai/utils/` - Enhanced Utilities

#### `hex_ai/utils/board_visualization.py`
- **Purpose**: Board visualization with policy overlays
- **Key Functions**:
  - `decode_policy_target()`: Decode one-hot policy to (row, col, trmph_move)
  - `visualize_board_with_policy()`: Display board with highlighted policy move
- **Dependencies**: `hex_ai.inference.board_display`, `hex_ai.utils.format_conversion`
- **Responsibilities**:
  - Policy move decoding
  - Board visualization with move highlighting
  - Integration with existing display system

#### `hex_ai/utils/consistency_checks.py`
- **Purpose**: Data consistency validation utilities
- **Key Functions**:
  - `policy_on_empty_cell()`: Check if policy move is on empty cell
  - `player_to_move_channel_valid()`: Validate player-to-move channel values
- **Dependencies**: `hex_ai.utils.format_conversion`
- **Responsibilities**:
  - Training data validation
  - Move consistency checking
  - Player-to-move validation

### 3. Enhanced Existing Files

#### `hex_ai/data_utils.py` (Enhanced)
- **New Functions Added**:
  - `load_examples_from_pkl()`: Load examples from .pkl.gz files
  - `extract_single_game_to_new_pkl()`: Extract individual games from processed files
- **Responsibilities**: Data loading and extraction utilities

#### `hex_ai/utils/__init__.py` (Enhanced)
- **New Exports**: Added exports for `board_visualization` and `consistency_checks` modules

## Files Removed

### Deleted Files
- `scripts/config.py` → moved to `hex_ai/trmph_processing/config.py`
- `scripts/processor.py` → moved to `hex_ai/trmph_processing/processor.py`
- `scripts/workers.py` → moved to `hex_ai/trmph_processing/workers.py`
- `scripts/lib/` (entire directory) → utilities moved to appropriate `hex_ai/` locations

### Modified Files
- `scripts/process_all_trmph.py` → simplified to CLI wrapper only
- `tests/test_multiprocessing_architecture.py` → updated imports
- Various deprecated scripts → updated imports

## Potential Duplication Analysis

### Board Visualization
- **`hex_ai/inference/board_display.py`**: Core board display functionality
- **`hex_ai/utils/board_visualization.py`**: Policy-aware visualization wrapper

**Analysis**: These serve different purposes:
- `board_display.py` provides core display functionality
- `board_visualization.py` provides policy-specific visualization features

**Recommendation**: Keep both - they have distinct responsibilities.

### Data Loading
- **`hex_ai/data_utils.py`**: Core data utilities
- **`hex_ai/trmph_processing/workers.py`**: Processing-specific data handling

**Analysis**: These serve different purposes:
- `data_utils.py` provides general data utilities
- `workers.py` provides processing-specific data handling

**Recommendation**: Keep both - they have distinct responsibilities.

## Entry Points

### Primary Entry Point
- **`scripts/process_all_trmph.py`**: Main CLI entry point for users

### Alternative Entry Points
- **`hex_ai/trmph_processing/cli.py`**: Direct CLI access
- **`hex_ai/trmph_processing/`**: Programmatic access

## Usage Patterns

### CLI Usage (Recommended)
```bash
python scripts/process_all_trmph.py --data-dir data --output-dir processed_data --max-workers 6
```

### Programmatic Usage
```python
from hex_ai.trmph_processing import ProcessingConfig, TRMPHProcessor

config = ProcessingConfig(data_dir="data", output_dir="processed_data", max_workers=6)
processor = TRMPHProcessor(config)
results = processor.process_all_files()
```

### Direct CLI Access
```python
from hex_ai.trmph_processing.cli import main
main()  # This would need sys.argv to be set properly
```

## Testing

All new files are covered by existing tests:
- `tests/test_multiprocessing_architecture.py` tests the core processing logic
- Import tests verify all modules can be imported correctly
- CLI tests verify the wrapper works correctly 