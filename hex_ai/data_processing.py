"""
Data processing utilities for Hex AI.

This module provides utilities for processing TRMPH game data into training-ready formats.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import gzip
import pickle
import numpy as np
from tqdm import tqdm

from .config import BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
from .data_utils import validate_game, extract_training_examples_from_game
from hex_ai.utils.format_conversion import parse_trmph_game_record
from hex_ai.value_utils import trmph_winner_to_training_value

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process raw .trmph files into efficient training formats."""
    
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def process_file(self, trmph_file: Path, file_idx: int, games_per_shard: int = 1000, 
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
        processed_games = self._convert_games_to_tensors(valid_games, file_idx)
        
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
    
    def _convert_games_to_tensors(self, games: List[Tuple[str, str]], file_idx: int) -> List[Dict]:
        """Convert games to dict format for training, including player_to_move."""
        processed_games = []
        for game_idx, (trmph_url, winner_indicator) in enumerate(tqdm(games, desc="Converting games to tensors")):
            try:
                game_id = (file_idx, game_idx+1)
                training_examples = extract_training_examples_from_game(trmph_url, winner_indicator, game_id)
                value_override = trmph_winner_to_training_value(winner_indicator)
                for example in training_examples:
                    board_state = example['board']
                    policy_target = example['policy']
                    value_target = example['value']
                    player_to_move = example['player_to_move']
                    if value_override is not None:
                        value_target = value_override
                    board_tensor = torch.FloatTensor(board_state)
                    if policy_target is not None:
                        policy_tensor = torch.FloatTensor(policy_target)
                    else:
                        policy_tensor = torch.zeros(POLICY_OUTPUT_SIZE, dtype=torch.float32)
                    value_tensor = torch.FloatTensor([value_target])
                    # Store as dict to include player_to_move
                    processed_games.append({
                        'board': board_tensor,
                        'policy': policy_tensor,
                        'value': value_tensor,
                        'player_to_move': player_to_move,
                        'metadata': example.get('metadata', {})
                    })
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
        
        for file_idx, trmph_file in enumerate(tqdm(trmph_files, desc="Processing files")):
            shard_files = self.process_file(trmph_file, file_idx, games_per_shard, compress)
            all_shard_files.extend(shard_files)
        
        logger.info(f"Total shard files created: {len(all_shard_files)}")
        return all_shard_files 