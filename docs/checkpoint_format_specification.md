# Checkpoint Format Specification

**Date:** 2024-12-19  
**Version:** 1.0

This document specifies the standard format for model checkpoints in the Hex AI project. All checkpoint saving and loading code should conform to this specification.

---

## Overview

Model checkpoints are saved as Python dictionaries using `torch.save()` and contain the complete state needed to resume training or perform inference. The format is designed to be consistent across all training and inference code.

---

## Standard Checkpoint Format

### Required Fields

All checkpoints must contain the following fields:

```python
checkpoint = {
    'model_state_dict': torch.nn.Module.state_dict(),  # Model weights and parameters
    'optimizer_state_dict': torch.optim.Optimizer.state_dict(),  # Optimizer state
    'epoch': int,  # Current training epoch (0-indexed)
    'best_val_loss': float,  # Best validation loss achieved so far
    'train_metrics': Dict[str, float],  # Training metrics for this checkpoint
    'val_metrics': Dict[str, float],  # Validation metrics for this checkpoint
    'mixed_precision': bool,  # Whether mixed precision was used
}
```

### Field Descriptions

#### `model_state_dict`
- **Type:** `torch.nn.Module.state_dict()`
- **Description:** Complete model state dictionary containing all weights, biases, and other learnable parameters
- **Required:** Yes
- **Example:** `{'conv1.weight': tensor(...), 'conv1.bias': tensor(...), ...}`

#### `optimizer_state_dict`
- **Type:** `torch.optim.Optimizer.state_dict()`
- **Description:** Complete optimizer state including momentum buffers, learning rate schedules, etc.
- **Required:** Yes
- **Example:** `{'state': {...}, 'param_groups': [...]}`

#### `epoch`
- **Type:** `int`
- **Description:** Current training epoch (0-indexed)
- **Required:** Yes
- **Example:** `5`

#### `best_val_loss`
- **Type:** `float`
- **Description:** Best validation loss achieved during training so far
- **Required:** Yes
- **Example:** `0.1234`

#### `train_metrics`
- **Type:** `Dict[str, float]`
- **Description:** Training metrics for the current checkpoint
- **Required:** Yes
- **Example:** `{'total_loss': 0.1234, 'policy_loss': 0.0567, 'value_loss': 0.0667}`

#### `val_metrics`
- **Type:** `Dict[str, float]` or `None`
- **Description:** Validation metrics for the current checkpoint (None if no validation data)
- **Required:** Yes
- **Example:** `{'total_loss': 0.1234, 'policy_loss': 0.0567, 'value_loss': 0.0667}`

#### `mixed_precision`
- **Type:** `bool`
- **Description:** Whether mixed precision training was used
- **Required:** Yes
- **Example:** `True`

---

## File Naming Convention

### Standard Checkpoints
- **Format:** `epoch{N}_mini{M}.pt.gz` (compressed) or `epoch{N}_mini{M}.pt` (uncompressed)
- **Example:** `epoch5_mini3.pt.gz`
- **Description:** Checkpoint from epoch N, mini-epoch M

### Best Model Checkpoints
- **Format:** `best_model.pt.gz` (compressed) or `best_model.pt` (uncompressed)
- **Description:** Checkpoint with the best validation loss

### Legacy Checkpoints
- **Format:** `checkpoint_epoch{N}.pt`
- **Example:** `checkpoint_epoch5.pt`
- **Description:** Legacy format (should be migrated to new format)

---

## Compression

### Supported Formats
- **Uncompressed:** `.pt` files (standard PyTorch format)
- **Gzipped:** `.pt.gz` files (compressed with gzip)

### Compression Guidelines
- **Default:** All new checkpoints are saved with gzip compression (`.pt.gz` format)
- **Compression Savings:** Typically 25-70% disk space reduction (27.7% average)
- **Backward Compatibility:** All loading code handles both compressed and uncompressed formats
- **Control:** Use `compress=True/False` parameter in save functions to control compression

---

## Loading Guidelines

### Standard Loading Pattern
```python
import torch
import gzip

def load_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load a checkpoint file."""
    # Check if file is gzipped
    with open(checkpoint_path, 'rb') as f:
        is_gzipped = f.read(2) == b'\x1f\x8b'
    
    # Load checkpoint
    if is_gzipped:
        with gzip.open(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(f, map_location=device, weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Validate format
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {type(checkpoint)}")
    
    required_keys = {'model_state_dict', 'optimizer_state_dict', 'epoch', 'best_val_loss', 
                    'train_metrics', 'val_metrics', 'mixed_precision'}
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")
    
    return checkpoint
```

### Error Handling
- Always use `weights_only=False` for loading (unless specifically migrating to weights-only format)
- Handle both gzipped and uncompressed files
- Provide clear error messages for format mismatches
- Validate checkpoint structure before using

---

## Implementation in Code

### Saving Checkpoints
All checkpoint saving should use the `Trainer.save_checkpoint()` method:

```python
def save_checkpoint(self, path: Path, train_metrics: Dict, val_metrics: Dict, compress: bool = True):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': self.current_epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_val_loss': self.best_val_loss,
        'mixed_precision': self.mixed_precision.use_mixed_precision
    }
    
    if compress:
        # Ensure path has .pt.gz extension
        if not str(path).endswith('.pt.gz'):
            path = path.with_suffix('.pt.gz')
        
        # Save as gzipped file
        import gzip
        with gzip.open(path, 'wb') as f:
            torch.save(checkpoint, f)
    else:
        # Save as uncompressed file
        torch.save(checkpoint, path)
```

### Loading Checkpoints
All checkpoint loading should use the `Trainer.load_checkpoint()` method:

```python
def load_checkpoint(self, path: Path):
    """Load model checkpoint."""
    # Check if file is gzipped by reading the first two bytes
    def is_gzipped(filepath):
        with open(filepath, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'
    
    if is_gzipped(path):
        import gzip
        with gzip.open(path, 'rb') as f:
            checkpoint = torch.load(f, map_location=self.device, weights_only=False)
    else:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.current_epoch = checkpoint['epoch']
    self.best_val_loss = checkpoint['best_val_loss']
```

---

## Validation

Use the checkpoint validation utility to verify checkpoint format:

```bash
# Validate a single checkpoint
python scripts/validate_checkpoints.py path/to/checkpoint.pt

# Validate all checkpoints in a directory
python scripts/validate_checkpoints.py --dir checkpoints/

# Audit all checkpoints in common directories
python scripts/validate_checkpoints.py --audit-all
```

---

## Migration from Legacy Formats

### Legacy Format Detection
- Check if checkpoint is a dictionary
- Look for required keys
- Handle missing keys gracefully

### Migration Strategy
1. Load legacy checkpoint
2. Extract available information
3. Create new format checkpoint with defaults for missing fields
4. Save in new format
5. Validate new checkpoint

---

## Best Practices

1. **Consistency:** Always use the same format for all checkpoints
2. **Validation:** Validate checkpoints after saving and before loading
3. **Error Handling:** Provide clear error messages for format issues
4. **Documentation:** Document any deviations from this specification
5. **Testing:** Test checkpoint saving/loading in CI/CD pipeline

---

## Future Considerations

- **Weights-only checkpoints:** Consider migrating to `weights_only=True` for inference-only checkpoints
- **Metadata expansion:** May add additional metadata fields as needed
- **Versioning:** Consider adding version field for format evolution
- **Compression alternatives:** May explore other compression formats (e.g., LZ4, Zstandard)

---

## References

- [PyTorch Save and Load Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Checkpoint Validation Utility](../scripts/validate_checkpoints.py)
- [Training Code](../hex_ai/training.py)
- [Model Wrapper](../hex_ai/inference/model_wrapper.py) 