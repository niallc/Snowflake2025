"""
Configuration management for TRMPH processing.

This module provides a clean configuration class that can be easily
serialized for multiprocessing and provides validation.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ProcessingConfig:
    """Configuration for TRMPH processing."""
    
    def __init__(self, 
                 data_dir: str, 
                 output_dir: str, 
                 max_files: Optional[int] = None,
                 position_selector: str = "all",
                 run_tag: Optional[str] = None,
                 max_workers: int = 6,
                 combine_output: bool = False):
        """
        Initialize processing configuration.
        
        Args:
            data_dir: Directory containing .trmph files
            output_dir: Output directory for processed files
            max_files: Maximum number of files to process (for testing)
            position_selector: Which positions to extract from each game
            run_tag: Tag for this processing run (default: timestamp)
            max_workers: Number of worker processes to use
            combine_output: Whether to create combined dataset after processing
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_files = max_files
        self.position_selector = position_selector
        self.run_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.max_workers = max_workers
        self.combine_output = combine_output
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for multiprocessing serialization."""
        return {
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'max_files': self.max_files,
            'position_selector': self.position_selector,
            'run_tag': self.run_tag,
            'max_workers': self.max_workers,
            'combine_output': self.combine_output
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProcessingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def validate(self) -> None:
        """Validate configuration and raise ValueError if invalid."""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        if self.position_selector not in ["all", "final", "penultimate"]:
            raise ValueError(f"Invalid position_selector: {self.position_selector}")
        
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {self.max_workers}")
        
        if self.max_files is not None and self.max_files < 1:
            raise ValueError(f"max_files must be at least 1, got {self.max_files}")
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (f"ProcessingConfig(data_dir='{self.data_dir}', "
                f"output_dir='{self.output_dir}', "
                f"max_files={self.max_files}, "
                f"position_selector='{self.position_selector}', "
                f"run_tag='{self.run_tag}', "
                f"max_workers={self.max_workers}, "
                f"combine_output={self.combine_output})") 