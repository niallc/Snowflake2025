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
from hex_ai.utils.format_conversion import parse_trmph_game_record, trmph_to_moves
from hex_ai.value_utils import trmph_winner_to_training_value
from hex_ai.enums import Player

logger = logging.getLogger(__name__)


def parse_trmph_line_flexible(line: str) -> Tuple[str, Optional[str]]:
    """
    Parse a TRMPH line that can be in any of these formats:
    1. Old format: "http://www.trmph.com/hex/board#13,moves b"
    2. New format: "#13,moves" (no winner indicator)
    3. Tournament format: "#13,moves winner" (with winner indicator)
    
    Args:
        line: TRMPH line to parse
        
    Returns:
        Tuple of (trmph_string, winner_indicator) where winner_indicator is None for new format
        
    Raises:
        ValueError: If the line format is invalid
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty line")
    
    # Handle new format: "#13,moves" (no winner indicator)
    if line.startswith('#13,'):
        # Remove any inline comments (everything after # that's not part of the game)
        if ' # ' in line:
            line = line.split(' # ')[0].strip()
        
        # Check if there's a winner indicator at the end
        parts = line.split()
        if len(parts) == 2:
            # Tournament format: "#13,moves winner"
            trmph_string, winner_indicator = parts
            if winner_indicator in ['b', 'r']:
                return trmph_string, winner_indicator
            else:
                raise ValueError(f"Invalid winner indicator '{winner_indicator}' in tournament format")
        elif len(parts) == 1:
            # New format: "#13,moves" (no winner indicator)
            return line, None
        else:
            raise ValueError(f"Invalid #13 format: expected 1 or 2 parts, got {len(parts)}")
    
    # Handle old format: "http://www.trmph.com/hex/board#13,moves b"
    elif line.startswith('http://www.trmph.com/hex/board#'):
        try:
            trmph_url, winner_indicator = parse_trmph_game_record(line)
            return trmph_url, winner_indicator
        except ValueError as e:
            raise ValueError(f"Invalid old format TRMPH line: {e}")
    
    else:
        raise ValueError(f"Unrecognized TRMPH format: {line}")


def extract_games_from_file_flexible(file_path: Path) -> List[Tuple[str, Optional[str]]]:
    """
    Extract game lines from a TRMPH file, handling both old and new formats.
    
    Args:
        file_path: Path to the TRMPH file
        
    Returns:
        List of tuples (trmph_string, winner_indicator) where winner_indicator is None for new format
    """
    games = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                trmph_string, winner_indicator = parse_trmph_line_flexible(line)
                games.append((trmph_string, winner_indicator))
            except ValueError as e:
                logger.warning(f"Skipping invalid line {line_num}: {e}")
                continue
    
    return games


def parse_trmph_to_gamerecord(trmph_string: str, winner_indicator: Optional[str] = None):
    """
    Parse a TRMPH string into a GameRecord object.
    
    Args:
        trmph_string: TRMPH format game string (with or without preamble)
        winner_indicator: Optional winner indicator ('b' or 'r')
        
    Returns:
        GameRecord object
    """
    # Import here to avoid circular imports
    from hex_ai.eval.strength_evaluator import GameRecord
    
    # Parse moves from TRMPH string
    moves = trmph_to_moves(trmph_string, BOARD_SIZE)
    
    # Convert to GameRecord format
    game_moves = []
    for i, (row, col) in enumerate(moves):
        player = Player.BLUE if i % 2 == 0 else Player.RED
        game_moves.append((row, col, player))
    
    return GameRecord(
        board_size=BOARD_SIZE,
        moves=game_moves,
        starting_player=Player.BLUE,
        metadata={"source": "trmph", "trmph_string": trmph_string, "winner_indicator": winner_indicator}
    )


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
    
    def _load_and_validate_games(self, file_path: Path) -> Tuple[List[Tuple[str, Optional[str]]], List[Tuple[str, Optional[str], str]]]:
        """Load and validate games from a .trmph file."""
        valid_games = []
        corrupted_games = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    trmph_string, winner_indicator = parse_trmph_line_flexible(line)
                except ValueError as e:
                    corrupted_games.append((line, "invalid_format", f"Invalid line format at line {line_num}: {e}"))
                    continue
                
                # For old format, validate trmph URL format
                if trmph_string.startswith("http://www.trmph.com/hex/board#") and winner_indicator is not None:
                    # Validate game integrity for old format
                    is_valid, error_msg = validate_game(trmph_string, winner_indicator, f"Line {line_num}")
                    if is_valid:
                        valid_games.append((trmph_string, winner_indicator))
                    else:
                        corrupted_games.append((trmph_string, winner_indicator, error_msg))
                elif trmph_string.startswith('#13,'):
                    # For new format, we can't validate without winner indicator, so just add it
                    valid_games.append((trmph_string, None))
                else:
                    corrupted_games.append((trmph_string, winner_indicator, f"Unrecognized trmph format at line {line_num}"))
        
        return valid_games, corrupted_games
    
    def _convert_games_to_tensors(self, games: List[Tuple[str, Optional[str]]], file_idx: int) -> List[Dict]:
        """Convert games to dict format for training, including player_to_move."""
        processed_games = []
        for game_idx, (trmph_url, winner_indicator) in enumerate(tqdm(games, desc="Converting games to tensors")):
            try:
                game_id = (file_idx, game_idx+1)
                # Skip games without winner indicator (new format) for now
                if winner_indicator is None:
                    logger.warning(f"Skipping game {game_id} - no winner indicator available")
                    continue
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