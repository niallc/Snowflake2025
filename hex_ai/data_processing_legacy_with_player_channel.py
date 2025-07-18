"""
Modified legacy data processing with player-to-move channel added.

This is a minimal modification of the legacy data processing to test whether
adding the player-to-move channel causes the performance regression.
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


class ProcessedDatasetLegacyWithPlayerChannel(torch.utils.data.Dataset):
    """
    Modified legacy dataset with player-to-move channel added.
    
    This is identical to the legacy ProcessedDataset except:
    - Adds player-to-move channel to the board state
    - Returns (batch_size, 3, 13, 13) instead of (batch_size, 2, 13, 13)
    
    Everything else remains exactly the same as the legacy dataset.
    """
    
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
                board_state = shard_data['boards'][local_idx]
                policy_target = shard_data['policies'][local_idx]
                value_target = shard_data['values'][local_idx]
                
                # MODIFIED: Add player-to-move channel
                board_np = board_state.numpy()
                from hex_ai.data_utils import get_player_to_move_from_board
                
                try:
                    player_to_move = get_player_to_move_from_board(board_np)
                except Exception as e:
                    # Use default value if we can't determine
                    from hex_ai.inference.board_utils import BLUE_PLAYER
                    player_to_move = BLUE_PLAYER
                
                # Create player-to-move channel
                player_channel = np.full((board_np.shape[1], board_np.shape[2]), float(player_to_move), dtype=np.float32)
                board_3ch = np.concatenate([board_np, player_channel[None, ...]], axis=0)
                board_state = torch.from_numpy(board_3ch)
                
                return board_state, policy_target, value_target
            
            current_idx += shard_size
        
        raise IndexError(f"Index {idx} out of range")


def create_processed_dataloader_legacy_with_player_channel(shard_files: List[Path], batch_size: int = 32,
                               shuffle: bool = True, num_workers: int = 4) -> torch.utils.data.DataLoader:
    """Create a DataLoader from processed shard files with player-to-move channel."""
    dataset = ProcessedDatasetLegacyWithPlayerChannel(shard_files, shuffle_shards=shuffle)
    
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