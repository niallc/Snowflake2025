"""
Data processing utilities for converting raw .trmph files to efficient training formats.

This module handles:
- Converting .trmph files to processed tensors
- Sharding data into manageable chunks
- Compressing data for efficient storage
- Creating training-ready datasets
"""

import torch
import gzip
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm

from .config import BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
from .data_utils import validate_game
from hex_ai.utils.format_conversion import parse_trmph_game_record

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process raw .trmph files into efficient training formats."""
    
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def process_file(self, trmph_file: Path, games_per_shard: int = 1000, 
                    compress: bool = True) -> List[Path]:
        """
        Process a single .trmph file into sharded, compressed training data.
        
        Args:
            trmph_file: Path to the .trmph file
            games_per_shard: Number of games per shard
            compress: Whether to compress the processed data
            
        Returns:
            List of paths to processed shard files
        """
        logger.info(f"Processing {trmph_file}")
        
        # Load and validate games
        valid_games, corrupted_games = self._load_and_validate_games(trmph_file)
        
        if not valid_games:
            logger.warning(f"No valid games found in {trmph_file}")
            return []
        
        # Convert games to tensors
        processed_games = self._convert_games_to_tensors(valid_games)
        
        # Shard the data
        shard_files = self._create_shards(processed_games, games_per_shard, compress)
        
        logger.info(f"Created {len(shard_files)} shards from {len(valid_games)} games")
        return shard_files
    
    def _load_and_validate_games(self, file_path: Path) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
        """Load and validate games from a .trmph file."""
        valid_games = []
        corrupted_games = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    trmph_url, winner_indicator = parse_trmph_game_record(line)
                except ValueError as e:
                    corrupted_games.append((line, "invalid_format", f"Invalid line format at line {line_num}: {e}"))
                    continue
                # Validate trmph URL format
                if not trmph_url.startswith("http://www.trmph.com/hex/board#"):
                    corrupted_games.append((trmph_url, winner_indicator, f"Invalid trmph URL format at line {line_num}"))
                    continue
                # Validate game integrity
                is_valid, error_msg = validate_game(trmph_url, winner_indicator, f"Line {line_num}")
                if is_valid:
                    valid_games.append((trmph_url, winner_indicator))
                else:
                    corrupted_games.append((trmph_url, winner_indicator, error_msg))
        
        return valid_games, corrupted_games
    
    def _convert_games_to_tensors(self, games: List[Tuple[str, str]]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Convert games to tensor format for training."""
        processed_games = []
        
        for trmph_url, winner_indicator in tqdm(games, desc="Converting games to tensors"):
            try:
                # Extract multiple training examples from each game
                from .data_utils import extract_training_examples_from_game
                training_examples = extract_training_examples_from_game(trmph_url, winner_indicator)
                
                # Override value targets based on actual winner
                if winner_indicator == "1":
                    value_override = 1.0  # Blue wins
                elif winner_indicator == "2":
                    value_override = 0.0  # Red wins
                else:
                    assert False, "Error (2nd) in data_processing.py, _convert_games_to_tensors: Unknown winner - cannot use game without a winner."
                
                for example in training_examples:
                    # Extract components from dictionary format
                    board_state = example['board']
                    policy_target = example['policy']
                    value_target = example['value']
                    
                    # Override value if we have explicit winner info
                    if value_override is not None:
                        value_target = value_override
                    
                    # Convert to tensors
                    board_tensor = torch.FloatTensor(board_state)
                    if policy_target is not None:
                        policy_tensor = torch.FloatTensor(policy_target)
                    else:
                        # Final positions have no next move to predict
                        # TODO(semi-urgent): Though we need to have some tensor here to make
                        #  training work, we should check whether all zeroes could lead to 
                        # infinite or NaN loss, which might confuse training.
                        # Hopefully not an issue as total_loss is to to value_loss only
                        # for final moves (the only time the policy label should be None), but
                        # given that any vector here should work, maybe a uniform label, 1/N^2 is safer.
                        policy_tensor = torch.zeros(POLICY_OUTPUT_SIZE, dtype=torch.float32)
                    value_tensor = torch.FloatTensor([value_target])
                    
                    processed_games.append((board_tensor, policy_tensor, value_tensor))
                
            except Exception as e:
                logger.warning(f"Failed to process game {trmph_url[:50]}...: {e}")
                continue
        
        return processed_games
    
    def _create_shards(self, processed_games: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
                       games_per_shard: int, compress: bool) -> List[Path]:
        """[REMOVED: This method is obsolete. All code now uses 'training_samples' format.]"""
        raise NotImplementedError("_create_shards is obsolete and should not be used.")
    
    def process_directory(self, trmph_dir: str, games_per_shard: int = 1000, 
                         compress: bool = True) -> List[Path]:
        """Process all .trmph files in a directory."""
        trmph_path = Path(trmph_dir)
        trmph_files = list(trmph_path.glob("*.trmph"))
        
        all_shard_files = []
        
        for trmph_file in tqdm(trmph_files, desc="Processing files"):
            shard_files = self.process_file(trmph_file, games_per_shard, compress)
            all_shard_files.extend(shard_files)
        
        logger.info(f"Total shard files created: {len(all_shard_files)}")
        return all_shard_files


# All code should use StreamingProcessedDataset from hex_ai.training_utils instead. 