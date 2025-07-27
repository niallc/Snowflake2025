"""
TRMPH Processing Module

This module provides multiprocessing capabilities for processing TRMPH files
into training-ready data formats.
"""

from .config import ProcessingConfig
from .processor import TRMPHProcessor, ParallelProcessor, SequentialProcessor
from .workers import process_single_file_worker, process_single_file_direct
from .cli import main

__all__ = [
    'ProcessingConfig',
    'TRMPHProcessor', 
    'ParallelProcessor',
    'SequentialProcessor',
    'process_single_file_worker',
    'process_single_file_direct',
    'main'
] 