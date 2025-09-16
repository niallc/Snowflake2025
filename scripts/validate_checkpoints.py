#!/usr/bin/env python3
"""
Checkpoint Validation Utility

This script validates checkpoint files to ensure they conform to the expected format
and can be loaded successfully. It helps identify inconsistencies in checkpoint saving
and loading across the codebase.

Usage:
    python scripts/validate_checkpoints.py <checkpoint_path>
    python scripts/validate_checkpoints.py --dir <checkpoint_directory>
    python scripts/validate_checkpoints.py --audit-all
"""

import argparse
import gzip
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import logging

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.models import TwoHeadedResNet
from hex_ai.training_utils import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointValidator:
    """Validates checkpoint files for format consistency and loadability."""
    
    def __init__(self):
        self.device = get_device()
        self.expected_keys = {
            'model_state_dict',
            'optimizer_state_dict', 
            'epoch',
            'best_val_loss',
            'train_metrics',
            'val_metrics',
            'mixed_precision'
        }
        
    def validate_single_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Validate a single checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'path': checkpoint_path,
            'exists': False,
            'is_gzipped': False,
            'can_load': False,
            'format_valid': False,
            'model_loadable': False,
            'errors': [],
            'warnings': [],
            'checkpoint_info': {}
        }
        
        path = Path(checkpoint_path)
        if not path.exists():
            result['errors'].append(f"File does not exist: {checkpoint_path}")
            return result
            
        result['exists'] = True
        
        # Check if file is gzipped
        try:
            with open(path, 'rb') as f:
                magic_bytes = f.read(2)
                result['is_gzipped'] = magic_bytes == b'\x1f\x8b'
        except Exception as e:
            result['errors'].append(f"Failed to read file: {e}")
            return result
            
        # Try to load the checkpoint
        try:
            if result['is_gzipped']:
                with gzip.open(path, 'rb') as f:
                    checkpoint = torch.load(f, map_location=self.device, weights_only=False)
            else:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            result['can_load'] = True
        except Exception as e:
            result['errors'].append(f"Failed to load checkpoint: {e}")
            return result
            
        # Validate checkpoint format
        if isinstance(checkpoint, dict):
            result['format_valid'] = True
            result['checkpoint_info'] = self._analyze_checkpoint_content(checkpoint)
            
            # Check for expected keys
            missing_keys = self.expected_keys - set(checkpoint.keys())
            if missing_keys:
                result['warnings'].append(f"Missing expected keys: {missing_keys}")
                
            # Check for unexpected keys
            unexpected_keys = set(checkpoint.keys()) - self.expected_keys
            if unexpected_keys:
                result['warnings'].append(f"Unexpected keys found: {unexpected_keys}")
        else:
            result['warnings'].append(f"Checkpoint is not a dictionary, type: {type(checkpoint)}")
            
        # Try to load model state dict
        if result['can_load'] and isinstance(checkpoint, dict):
            try:
                if 'model_state_dict' in checkpoint:
                    model = TwoHeadedResNet()
                    model.load_state_dict(checkpoint['model_state_dict'])
                    result['model_loadable'] = True
                else:
                    result['warnings'].append("No 'model_state_dict' found in checkpoint")
            except Exception as e:
                result['errors'].append(f"Failed to load model state dict: {e}")
                
        return result
        
    def _analyze_checkpoint_content(self, checkpoint: Dict) -> Dict[str, Any]:
        """Analyze the content of a checkpoint dictionary."""
        info = {}
        
        # Basic info
        info['epoch'] = checkpoint.get('epoch', 'Not found')
        info['best_val_loss'] = checkpoint.get('best_val_loss', 'Not found')
        info['mixed_precision'] = checkpoint.get('mixed_precision', 'Not found')
        
        # Model state dict info
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            info['model_state_dict_keys'] = len(model_state.keys())
            info['model_state_dict_types'] = {k: type(v).__name__ for k, v in list(model_state.items())[:5]}
            
        # Optimizer state dict info
        if 'optimizer_state_dict' in checkpoint:
            opt_state = checkpoint['optimizer_state_dict']
            info['optimizer_state_dict_keys'] = len(opt_state.keys())
            
        # Metrics info
        if 'train_metrics' in checkpoint:
            train_metrics = checkpoint['train_metrics']
            info['train_metrics_keys'] = list(train_metrics.keys()) if isinstance(train_metrics, dict) else 'Not a dict'
            
        if 'val_metrics' in checkpoint:
            val_metrics = checkpoint['val_metrics']
            info['val_metrics_keys'] = list(val_metrics.keys()) if isinstance(val_metrics, dict) else 'Not a dict'
            
        return info
        
    def validate_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Validate all checkpoint files in a directory.
        
        Args:
            directory_path: Path to directory containing checkpoints
            
        Returns:
            List of validation results for each checkpoint
        """
        results = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return results
            
        # Find all checkpoint files (including subdirectories)
        checkpoint_files = []
        for pattern in ['*.pt', '*.pt.gz']:
            checkpoint_files.extend(directory.rglob(pattern))
            
        logger.info(f"Found {len(checkpoint_files)} checkpoint files in {directory_path}")
        
        for checkpoint_file in checkpoint_files:
            logger.info(f"Validating {checkpoint_file}")
            result = self.validate_single_checkpoint(str(checkpoint_file))
            results.append(result)
            
        return results
        
    def audit_all_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Audit all checkpoints in common checkpoint directories.
        
        Returns:
            List of validation results for all found checkpoints
        """
        common_dirs = [
            "checkpoints",
            "checkpoints/hyperparameter_tuning",
            "checkpoints/scaled_tuning"
        ]
        
        all_results = []
        for directory in common_dirs:
            if Path(directory).exists():
                logger.info(f"Auditing directory: {directory}")
                results = self.validate_directory(directory)
                all_results.extend(results)
                
        return all_results
        
    def print_validation_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of validation results."""
        if not results:
            print("No checkpoints found to validate.")
            return
            
        total = len(results)
        valid = sum(1 for r in results if r['can_load'] and r['format_valid'])
        loadable = sum(1 for r in results if r['model_loadable'])
        
        print(f"\n=== Checkpoint Validation Summary ===")
        print(f"Total checkpoints: {total}")
        print(f"Can load: {valid}")
        print(f"Model loadable: {loadable}")
        print(f"Success rate: {valid/total*100:.1f}%")
        
        # Show errors and warnings
        all_errors = []
        all_warnings = []
        for result in results:
            all_errors.extend(result['errors'])
            all_warnings.extend(result['warnings'])
            
        if all_errors:
            print(f"\n=== Errors ({len(all_errors)} total) ===")
            for error in all_errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(all_errors) > 10:
                print(f"  ... and {len(all_errors) - 10} more errors")
                
        if all_warnings:
            print(f"\n=== Warnings ({len(all_warnings)} total) ===")
            for warning in all_warnings[:10]:  # Show first 10 warnings
                print(f"  - {warning}")
            if len(all_warnings) > 10:
                print(f"  ... and {len(all_warnings) - 10} more warnings")
                
        # Show detailed results for failed checkpoints
        failed = [r for r in results if not r['can_load'] or not r['format_valid']]
        if failed:
            print(f"\n=== Failed Checkpoints ===")
            for result in failed:
                print(f"\n{result['path']}:")
                for error in result['errors']:
                    print(f"  ERROR: {error}")
                for warning in result['warnings']:
                    print(f"  WARNING: {warning}")


def main():
    parser = argparse.ArgumentParser(description="Validate checkpoint files")
    parser.add_argument("checkpoint_path", nargs="?", help="Path to a single checkpoint file")
    parser.add_argument("--dir", help="Directory containing checkpoints to validate")
    parser.add_argument("--audit-all", action="store_true", help="Audit all checkpoints in common directories")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    validator = CheckpointValidator()
    
    if args.audit_all:
        results = validator.audit_all_checkpoints()
    elif args.dir:
        results = validator.validate_directory(args.dir)
    elif args.checkpoint_path:
        result = validator.validate_single_checkpoint(args.checkpoint_path)
        results = [result]
    else:
        parser.print_help()
        return
        
    validator.print_validation_summary(results)


if __name__ == "__main__":
    main() 