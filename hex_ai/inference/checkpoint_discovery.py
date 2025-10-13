"""
Checkpoint discovery and ordering system.

This module handles discovering and ordering model checkpoints from directories
for use in knockout tournaments.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a discovered checkpoint."""
    file_path: Path
    epoch: int
    mini: int
    total_mini_epochs: int
    creation_time: float
    
    @property
    def checkpoint_number(self) -> int:
        """Get the sequential checkpoint number (1-indexed)."""
        return (self.epoch - 1) * self.total_mini_epochs + self.mini + 1
    
    @property
    def name(self) -> str:
        """Get a human-readable name for this checkpoint."""
        return f"epoch{self.epoch}_mini{self.mini}"


class CheckpointDiscovery:
    """
    Discovers and orders model checkpoints from a directory.
    
    Expects checkpoint files with the pattern: epochN_miniJ.pt.gz
    where N is the epoch number and J is the mini-epoch number.
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint discovery for a directory.
        
        Args:
            checkpoint_dir: Path to directory containing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {checkpoint_dir}. "
                f"Please check the path and ensure the directory exists."
            )
        if not self.checkpoint_dir.is_dir():
            raise ValueError(
                f"Path is not a directory: {checkpoint_dir}. "
                f"Please provide a directory path, not a file path."
            )
        
        self.checkpoint_pattern = re.compile(r'epoch(\d+)_mini(\d+)\.pt\.gz$')
        self._discovered_checkpoints: Optional[List[CheckpointInfo]] = None
    
    def discover_checkpoints(self) -> List[CheckpointInfo]:
        """
        Discover all checkpoints in the directory.
        
        Returns:
            List of CheckpointInfo objects, ordered by creation time
        """
        if self._discovered_checkpoints is not None:
            return self._discovered_checkpoints
        
        checkpoints = []
        
        # Find all matching files
        for file_path in self.checkpoint_dir.iterdir():
            if file_path.is_file():
                match = self.checkpoint_pattern.match(file_path.name)
                if match:
                    epoch = int(match.group(1))
                    mini = int(match.group(2))
                    
                    # Get file creation time
                    creation_time = file_path.stat().st_mtime
                    
                    checkpoint = CheckpointInfo(
                        file_path=file_path,
                        epoch=epoch,
                        mini=mini,
                        total_mini_epochs=0,  # Will be set later
                        creation_time=creation_time
                    )
                    checkpoints.append(checkpoint)
        
        if not checkpoints:
            raise ValueError(
                f"No checkpoint files found in {self.checkpoint_dir}. "
                f"Expected files matching pattern 'epochN_miniJ.pt.gz' (e.g., epoch1_mini1.pt.gz). "
                f"Please check that the directory contains valid checkpoint files."
            )
        
        # Determine total mini epochs per epoch
        max_epoch = max(cp.epoch for cp in checkpoints)
        max_mini_per_epoch = {}
        for cp in checkpoints:
            if cp.epoch not in max_mini_per_epoch:
                max_mini_per_epoch[cp.epoch] = cp.mini
            else:
                max_mini_per_epoch[cp.epoch] = max(max_mini_per_epoch[cp.epoch], cp.mini)
        
        # Set total mini epochs (all completed epochs must have the same number of mini epochs)
        # The most recent epoch is allowed to have fewer mini epochs (incomplete training)
        if len(max_mini_per_epoch) > 1:
            # Check all epochs except the most recent one
            sorted_epochs = sorted(max_mini_per_epoch.keys())
            completed_epochs = sorted_epochs[:-1]  # All except the most recent
            
            if len(completed_epochs) > 1:
                completed_mini_counts = [max_mini_per_epoch[epoch] for epoch in completed_epochs]
                if len(set(completed_mini_counts)) > 1:
                    raise ValueError(
                        f"Completed epochs have different numbers of mini epochs: {dict(zip(completed_epochs, completed_mini_counts))}. "
                        f"This indicates inconsistent training configuration and is not supported."
                    )
        
        total_mini_epochs = max(max_mini_per_epoch.values())
        
        # Update checkpoint info
        for cp in checkpoints:
            cp.total_mini_epochs = total_mini_epochs
        
        # Sort by creation time to ensure proper ordering
        checkpoints.sort(key=lambda cp: cp.creation_time)
        
        # Validate ordering (fail fast on invalid state)
        # DISABLED: Allow non-sequential checkpoints for tournament flexibility
        # self._validate_checkpoint_ordering(checkpoints)
        
        self._discovered_checkpoints = checkpoints
        logger.info(f"Discovered {len(checkpoints)} checkpoints in {self.checkpoint_dir}")
        
        return checkpoints
    
    def _validate_checkpoint_ordering(self, checkpoints: List[CheckpointInfo]):
        """Validate that checkpoints are in the expected order."""
        for i in range(1, len(checkpoints)):
            prev_cp = checkpoints[i-1]
            curr_cp = checkpoints[i]
            
            # Check that checkpoint numbers are sequential
            expected_number = prev_cp.checkpoint_number + 1
            if curr_cp.checkpoint_number != expected_number:
                raise ValueError(
                    f"Non-sequential checkpoint numbers detected: {prev_cp.name} (#{prev_cp.checkpoint_number}) "
                    f"followed by {curr_cp.name} (#{curr_cp.checkpoint_number}), expected #{expected_number}. "
                    f"This indicates missing or corrupted checkpoints in the training sequence."
                )
    
    def get_checkpoint_by_number(self, checkpoint_number: int) -> CheckpointInfo:
        """
        Get a checkpoint by its sequential number.
        
        Args:
            checkpoint_number: 1-indexed checkpoint number
            
        Returns:
            CheckpointInfo for the specified checkpoint
        """
        checkpoints = self.discover_checkpoints()
        
        for cp in checkpoints:
            if cp.checkpoint_number == checkpoint_number:
                return cp
        
        raise ValueError(
            f"Checkpoint #{checkpoint_number} not found. "
            f"Available checkpoints: {[cp.checkpoint_number for cp in checkpoints]}. "
            f"Please check the checkpoint number or ensure the checkpoint exists."
        )
    
    def get_checkpoints_by_range(self, start: int, end: int) -> List[CheckpointInfo]:
        """
        Get checkpoints within a range of sequential numbers.
        
        Args:
            start: Start checkpoint number (inclusive, 1-indexed)
            end: End checkpoint number (exclusive, 1-indexed)
            
        Returns:
            List of CheckpointInfo objects in the specified range
        """
        checkpoints = self.discover_checkpoints()
        
        result = []
        for cp in checkpoints:
            if start <= cp.checkpoint_number < end:
                result.append(cp)
        
        return result
    
    def get_latest_checkpoints(self, count: int) -> List[CheckpointInfo]:
        """
        Get the latest N checkpoints.
        
        Args:
            count: Number of latest checkpoints to return
            
        Returns:
            List of the latest CheckpointInfo objects
        """
        checkpoints = self.discover_checkpoints()
        return checkpoints[-count:] if count <= len(checkpoints) else checkpoints
    
    def get_checkpoints_by_epoch_range(self, start_epoch: int, end_epoch: int) -> List[CheckpointInfo]:
        """
        Get checkpoints within a range of epoch numbers.
        
        Args:
            start_epoch: Start epoch number (inclusive)
            end_epoch: End epoch number (exclusive)
            
        Returns:
            List of CheckpointInfo objects in the specified epoch range
        """
        checkpoints = self.discover_checkpoints()
        
        result = []
        for cp in checkpoints:
            if start_epoch <= cp.epoch < end_epoch:
                result.append(cp)
        
        return result
    
    def get_checkpoints_by_mini_epoch_range(self, start_mini_epoch: int, end_mini_epoch: int) -> List[CheckpointInfo]:
        """
        Get checkpoints within a range of mini epoch numbers.
        
        Args:
            start_mini_epoch: Start mini epoch number (inclusive)
            end_mini_epoch: End mini epoch number (exclusive)
            
        Returns:
            List of CheckpointInfo objects in the specified mini epoch range
        """
        checkpoints = self.discover_checkpoints()
        
        result = []
        for cp in checkpoints:
            if start_mini_epoch <= cp.mini < end_mini_epoch:
                result.append(cp)
        
        return result
    
    def get_checkpoints_by_combined_range(self, epoch_range: Optional[Tuple[int, int]] = None, 
                                        mini_epoch_range: Optional[Tuple[int, int]] = None) -> List[CheckpointInfo]:
        """
        Get checkpoints filtered by both epoch and mini epoch ranges.
        
        Args:
            epoch_range: Optional tuple of (start_epoch, end_epoch) to filter by epoch
            mini_epoch_range: Optional tuple of (start_mini_epoch, end_mini_epoch) to filter by mini epoch
            
        Returns:
            List of CheckpointInfo objects matching both criteria
        """
        checkpoints = self.discover_checkpoints()
        
        result = []
        for cp in checkpoints:
            # Check epoch range if specified
            if epoch_range:
                start_epoch, end_epoch = epoch_range
                if not (start_epoch <= cp.epoch < end_epoch):
                    continue
            
            # Check mini epoch range if specified
            if mini_epoch_range:
                start_mini_epoch, end_mini_epoch = mini_epoch_range
                if not (start_mini_epoch <= cp.mini < end_mini_epoch):
                    continue
            
            result.append(cp)
        
        return result
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get a summary of discovered checkpoints."""
        checkpoints = self.discover_checkpoints()
        
        if not checkpoints:
            return {"total_checkpoints": 0}
        
        epochs = set(cp.epoch for cp in checkpoints)
        min_epoch, max_epoch = min(epochs), max(epochs)
        
        # Calculate mini epochs per epoch for summary
        max_mini_per_epoch = {}
        for cp in checkpoints:
            if cp.epoch not in max_mini_per_epoch:
                max_mini_per_epoch[cp.epoch] = cp.mini
            else:
                max_mini_per_epoch[cp.epoch] = max(max_mini_per_epoch[cp.epoch], cp.mini)
        
        return {
            "total_checkpoints": len(checkpoints),
            "epoch_range": (min_epoch, max_epoch),
            "total_epochs": len(epochs),
            "mini_epochs_per_epoch": checkpoints[0].total_mini_epochs,
            "mini_epochs_by_epoch": max_mini_per_epoch,
            "first_checkpoint": checkpoints[0].name,
            "last_checkpoint": checkpoints[-1].name,
            "checkpoint_numbers": [cp.checkpoint_number for cp in checkpoints]
        }
