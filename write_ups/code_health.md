# Code Health Review: Hex AI Project

**Date:** 2024-07-17

This document summarizes code health issues, technical debt, and recommendations for the Hex AI project, based on recent debugging and development work. It is intended as a living document to guide future refactoring and cleanup.

---

## 1. Model Checkpointing and Loading

### Issues
- **Inconsistent checkpoint formats:**
  - Some checkpoints (e.g., `best_model.pt`) are full dicts with model, optimizer, and metadata.
  - Others (e.g., `checkpoint_epoch_0.pt`) may be partial, weights-only, or saved with different settings.
  - Some files are gzipped, some may not be, and the naming convention does not always make this clear.
- **Loader logic is complex and brittle:**
  - Loader tries to handle both gzipped and uncompressed, weights-only and full dicts, and both new and legacy models.
  - Error messages can be long and confusing, especially with PyTorch 2.6+ and the `weights_only` change.
  - Loader does not always clearly distinguish between file corruption, format mismatch, and unsupported pickle protocols.
- **Lack of documentation:**
  - There is no single place documenting the expected checkpoint format, compression, or how to migrate old checkpoints.

### Recommendations
- **Standardize checkpoint saving:**
  - Always save checkpoints in the same format (full dict, gzipped or not, with clear naming).
  - Document the format and update all save/load code to match.
- **Simplify loader logic:**
  - Check if the file is gzipped by magic bytes, then always load with `weights_only=False` (unless you migrate to weights-only checkpoints in the future).
  - If the loaded object is a dict with `'model_state_dict'`, use it; if it's a plain state dict, load it directly.
  - Print concise, actionable error messages.
- **Add migration/validation scripts:**
  - Provide a script to check, validate, and (if needed) convert old checkpoints to the new standard.
- **Document everything:**
  - Add a section to the README and/or this file describing the checkpoint format, how to save/load, and how to handle legacy files.

---

## 2. Legacy Model Support

### Issues
- Legacy model support is currently handled via a separate class and CLI flag, which is good for separation but adds complexity.
- There is a risk of confusion if legacy code is not clearly marked and eventually removed.

### Recommendations
- Keep legacy code isolated and well-documented.
- Add TODOs and a clear removal plan for legacy support after migration/validation.
- Consider writing a migration script to convert old checkpoints to the new format, then remove legacy code.

---

## 3. Error Handling and Diagnostics

### Issues
- Error messages from model loading can be verbose and hard to interpret, especially with fallback logic.
- There is no clear distinction between file corruption, format mismatch, and unsupported features.

### Recommendations
- Print short, clear error messages with actionable advice.
- Log (or print) the first few bytes of a file if loading fails, to help diagnose format issues.
- Consider adding a utility script to check and summarize checkpoint files.

---

## 4. Documentation and Code Organization

### Issues
- Some scripts and modules lack clear docstrings or usage examples.
- The distinction between library code, script-level utilities, and legacy code could be clearer.

### Recommendations
- Add/expand docstrings and usage examples for all scripts and modules.
- Keep `hex_ai/` for core library code, `scripts/` for CLI and analysis tools, and `scripts/lib/` for script-level utilities.
- Regularly review and clean up unused or deprecated code.

---

## 5. Testing and Reproducibility

### Issues
- There is limited automated testing for data loading, model instantiation, and inference.
- Reproducibility of experiments depends on careful manual tracking of configs and seeds.

### Recommendations
- Add basic tests for data loading, model forward pass, and checkpoint loading.
- Save all configs and random seeds with each experiment.
- Consider using a config management tool or experiment tracker for larger runs.

---

## 6. Miscellaneous

- Ensure all dependencies are listed in `requirements.txt` and are up to date.
- Add a section to the README about activating the virtual environment and installing dependencies.
- Consider adding a script to check environment and dependency versions.

---

**This document is a starting point. Please add, update, and refine as the project evolves!** 