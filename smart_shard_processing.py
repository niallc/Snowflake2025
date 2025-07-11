#!/usr/bin/env python3
"""
Smart shard processing with _i suffixes to avoid overwriting.
"""

import torch
import gzip
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import gc

from hex_ai.config import BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
from hex_ai.data_utils import convert_to_matrix_format

logger = logging.getLogger(__name__)


def process_single_game(args: Tuple[str, str]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Process a single game - optimized to do conversion only once."""
    trmph_url, winner_indicator = args
    
    try:
        # Convert to matrix format (ONCE - not twice like before)
        board_state, policy_target, value_target = convert_to_matrix_format(trmph_url)
        
        # Override value target based on actual winner
        if winner_indicator == "1":
            value_target = 1.0  # Blue wins
        elif winner_indicator == "0":
            value_target = 0.0  # Red wins
        
        # Convert to tensors
        board_tensor = torch.FloatTensor(board_state)
        policy_tensor = torch.FloatTensor(policy_target)
        value_tensor = torch.FloatTensor([value_target])
        
        return (board_tensor, policy_tensor, value_tensor)
        
    except Exception as e:
        # Silently skip corrupted games instead of logging each one
        return None


def load_games_from_file_fast(file_path: Path) -> List[Tuple[str, str]]:
    """Load games from file with minimal validation."""
    games = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Parse line format: "trmph_url winner_indicator"
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            
            trmph_url, winner_indicator = parts
            
            # Basic URL format check only
            if not trmph_url.startswith("http://www.trmph.com/hex/board#"):
                continue
            
            games.append((trmph_url, winner_indicator))
    
    return games


def get_next_shard_filename(processed_dir: Path, shard_idx: int) -> Path:
    """Get the next available shard filename with _i suffix if needed."""
    base_name = f"shard_{shard_idx:04d}.pkl.gz"
    base_path = processed_dir / base_name
    
    if not base_path.exists():
        return base_path
    
    # Find the next available suffix
    counter = 1
    while True:
        suffixed_name = f"shard_{shard_idx:04d}_{counter:02d}.pkl.gz"
        suffixed_path = processed_dir / suffixed_name
        if not suffixed_path.exists():
            return suffixed_path
        counter += 1


def process_file_smart(trmph_file: Path, processed_dir: Path, games_per_shard: int = 1000, 
                      compress: bool = True, num_workers: int = None) -> List[Path]:
    """Process a single file with smart shard naming."""
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Cap at 8 workers
    
    # Ensure output directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {trmph_file} with {num_workers} workers")
    
    # Load games (fast - no validation)
    games = load_games_from_file_fast(trmph_file)
    
    if not games:
        logger.warning(f"No games found in {trmph_file}")
        return []
    
    # Process games in parallel
    processed_games = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all games for processing
        future_to_game = {executor.submit(process_single_game, game): game for game in games}
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_game), total=len(games), 
                          desc=f"Processing {trmph_file.name}"):
            result = future.result()
            if result is not None:
                processed_games.append(result)
    
    # Create shards with smart naming
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
            'num_games': len(shard_games)
        }
        
        # Get next available shard filename
        shard_file = get_next_shard_filename(processed_dir, shard_idx // games_per_shard)
        
        # Save shard
        with gzip.open(shard_file, 'wb') as f:
            pickle.dump(shard_data, f)
        
        shard_files.append(shard_file)
    
    logger.info(f"Created {len(shard_files)} shards from {len(processed_games)} games")
    return shard_files


def process_all_data_global_shards(
    source_dir: str = "data/twoNetGames",
    processed_dir: str = "data/processed",
    games_per_shard: int = 1000,
    compress: bool = True,
    num_workers: int = None
) -> List[Path]:
    """Process all data with global sharding and monotonic shard numbering."""
    source_path = Path(source_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    # Find all .trmph files
    trmph_files = list(source_path.glob("*.trmph"))
    trmph_files.sort()  # Ensure consistent ordering

    if not trmph_files:
        raise FileNotFoundError(f"No .trmph files found in {source_dir}")

    logger.info(f"Found {len(trmph_files)} files to process")
    logger.info(f"Using {num_workers} workers for parallel processing")
    logger.info("Using global shard numbering to avoid overwrites and ensure monotonicity")

    start_time = time.time()
    all_shard_files = []
    global_game_buffer = []
    global_shard_idx = 0
    total_games = 0

    # Read all games from all files into a global buffer
    for i, trmph_file in enumerate(trmph_files, 1):
        logger.info(f"Loading games from file {i}/{len(trmph_files)}: {trmph_file.name}")
        games = load_games_from_file_fast(trmph_file)
        if not games:
            logger.warning(f"No games found in {trmph_file}")
            continue
        global_game_buffer.extend(games)
        total_games += len(games)

        # While enough games for a shard, process and write
        while len(global_game_buffer) >= games_per_shard:
            shard_games = global_game_buffer[:games_per_shard]
            global_game_buffer = global_game_buffer[games_per_shard:]

            # Process games in parallel
            processed_games = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_game = {executor.submit(process_single_game, game): game for game in shard_games}
                for future in tqdm(as_completed(future_to_game), total=len(shard_games), desc=f"Processing shard {global_shard_idx:04d}"):
                    result = future.result()
                    if result is not None:
                        processed_games.append(result)

            if not processed_games:
                logger.warning(f"No valid games in shard {global_shard_idx:04d}")
                continue

            # Create shard data
            boards = torch.stack([game[0] for game in processed_games])
            policies = torch.stack([game[1] for game in processed_games])
            values = torch.stack([game[2] for game in processed_games])
            shard_data = {
                'boards': boards,
                'policies': policies,
                'values': values,
                'num_games': len(processed_games)
            }
            shard_file = processed_path / f"shard_{global_shard_idx:05d}.pkl.gz"
            with gzip.open(shard_file, 'wb') as f:
                pickle.dump(shard_data, f)
            logger.info(f"Created shard {shard_file} with {len(processed_games)} games")
            all_shard_files.append(shard_file)
            global_shard_idx += 1
            # Explicitly free memory
            del processed_games, boards, policies, values, shard_data
            gc.collect()

    # Process any remaining games (final shard)
    if global_game_buffer:
        logger.info(f"Processing final shard with {len(global_game_buffer)} games")
        processed_games = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_game = {executor.submit(process_single_game, game): game for game in global_game_buffer}
            for future in tqdm(as_completed(future_to_game), total=len(global_game_buffer), desc=f"Processing final shard {global_shard_idx:04d}"):
                result = future.result()
                if result is not None:
                    processed_games.append(result)
        if processed_games:
            boards = torch.stack([game[0] for game in processed_games])
            policies = torch.stack([game[1] for game in processed_games])
            values = torch.stack([game[2] for game in processed_games])
            shard_data = {
                'boards': boards,
                'policies': policies,
                'values': values,
                'num_games': len(processed_games)
            }
            shard_file = processed_path / f"shard_{global_shard_idx:05d}.pkl.gz"
            with gzip.open(shard_file, 'wb') as f:
                pickle.dump(shard_data, f)
            logger.info(f"Created final shard {shard_file} with {len(processed_games)} games")
            all_shard_files.append(shard_file)
            global_shard_idx += 1
            # Explicitly free memory
            del processed_games, boards, policies, values, shard_data
            gc.collect()

    processing_time = time.time() - start_time
    final_shards = list(processed_path.glob("*.pkl.gz"))
    logger.info(f"\n{'='*60}")
    logger.info("GLOBAL SHARD PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {len(trmph_files)}")
    logger.info(f"Total games processed: {total_games}")
    logger.info(f"New shards created: {len(all_shard_files)}")
    logger.info(f"Total shards now: {len(final_shards)}")
    logger.info(f"Processing time: {processing_time:.1f} seconds")
    if processing_time > 0:
        logger.info(f"Games per second: {total_games / processing_time:.1f}")
    logger.info(f"Processed data location: {processed_path}")
    logger.info(f"{'='*60}")
    return all_shard_files


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Process all data with global shard numbering
    process_all_data_global_shards() 