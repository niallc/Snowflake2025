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
                
                # Parse line format: "trmph_url winner_indicator"
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    corrupted_games.append((line, "invalid_format", f"Invalid line format at line {line_num}"))
                    continue
                
                trmph_url, winner_indicator = parts
                
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
                training_examples = extract_training_examples_from_game(trmph_url, f"Training data - Game from {winner_indicator}")
                
                # Override value targets based on actual winner
                if winner_indicator == "1":
                    value_override = 1.0  # Blue wins
                elif winner_indicator == "2":
                    value_override = 0.0  # Red wins
                else:
                    assert False, "Error (2nd) in data_processing.py, _convert_games_to_tensors: Unknown winner - cannot use game without a winner."
                
                for board_state, policy_target, value_target in training_examples:
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
        """Create sharded files from processed games."""
        shard_files = []
        
        for shard_idx in range(0, len(processed_games), games_per_shard):
            shard_games = processed_games[shard_idx:shard_idx + games_per_shard]
            
            # Create shard data
            boards = torch.stack([game[0] for game in shard_games])
            policies = torch.stack([game[1] for game in shard_games])
            values = torch.stack([game[2] for game in shard_games])
            
            shard_data = {
                'boards': boards,
                'policies': policies,
                'values': values,
                'num_games': len(shard_games)  # Now represents training examples, not games
            }
            
            # Save shard
            shard_file = self.processed_dir / f"shard_{shard_idx//games_per_shard:04d}.pkl"
            if compress:
                shard_file = shard_file.with_suffix('.pkl.gz')
                with gzip.open(shard_file, 'wb') as f:
                    pickle.dump(shard_data, f)
            else:
                with open(shard_file, 'wb') as f:
                    pickle.dump(shard_data, f)
            
            shard_files.append(shard_file)
        
        return shard_files
    
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


class ProcessedDataset(torch.utils.data.Dataset):
    """Dataset for loading processed shard files."""
    
    def __init__(self, shard_files: List[Path], shuffle_shards: bool = True):
        self.shard_files = shard_files
        if shuffle_shards:
            np.random.shuffle(self.shard_files)
        
        # Calculate total number of games
        self.total_games = 0
        for shard_file in self.shard_files:
            if shard_file.suffix == '.gz':
                with gzip.open(shard_file, 'rb') as f:
                    shard_data = pickle.load(f)
            else:
                with open(shard_file, 'rb') as f:
                    shard_data = pickle.load(f)
            self.total_games += shard_data['num_games']
        
        logger.info(f"Loaded {len(self.shard_files)} shards with {self.total_games} total games")
    
    def __len__(self):
        return self.total_games
    
    def __getitem__(self, idx):
        # Find which shard contains this index
        current_idx = 0
        for shard_file in self.shard_files:
            if shard_file.suffix == '.gz':
                with gzip.open(shard_file, 'rb') as f:
                    shard_data = pickle.load(f)
            else:
                with open(shard_file, 'rb') as f:
                    shard_data = pickle.load(f)
            
            shard_size = shard_data['num_games']
            if current_idx <= idx < current_idx + shard_size:
                # Found the right shard
                local_idx = idx - current_idx
                return (shard_data['boards'][local_idx], 
                       shard_data['policies'][local_idx], 
                       shard_data['values'][local_idx])
            
            current_idx += shard_size
        
        raise IndexError(f"Index {idx} out of range")


def create_processed_dataloader(shard_files: List[Path], batch_size: int = 32,
                               shuffle: bool = True, num_workers: int = 4) -> torch.utils.data.DataLoader:
    """Create a DataLoader from processed shard files."""
    dataset = ProcessedDataset(shard_files, shuffle_shards=shuffle)
    
    # Disable pin_memory for MPS (not supported)
    import torch
    pin_memory = torch.backends.mps.is_available() == False  # Only enable for non-MPS devices
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    ) 