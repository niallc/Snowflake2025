#!/usr/bin/env python3
"""
Safe shard processing with memory monitoring and emergency shutdown.
"""

import torch
import gzip
import pickle
import logging
import psutil
import os
import signal
import sys
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

# Safety configuration
MAX_MEMORY_PERCENT = 80  # Exit if memory usage exceeds 80%
MEMORY_CHECK_INTERVAL = 5000  # Check memory every 5000 games processed
MAX_WORKERS = 4  # Reduced from 8 to 4
EMERGENCY_SHUTDOWN = False  # Global flag for emergency shutdown

def log_memory_usage(context: str = ""):
    """Log memory usage for main process and all children."""
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 * 1024)
        logger.info(f"[{context}] Main process memory: {mem:.2f} MB")
        
        # Log all children
        children = process.children(recursive=True)
        if children:
            total_child_mem = 0
            for child in children:
                cmem = child.memory_info().rss / (1024 * 1024)
                total_child_mem += cmem
                logger.info(f"[{context}] Child PID {child.pid} memory: {cmem:.2f} MB")
            logger.info(f"[{context}] Total child memory: {total_child_mem:.2f} MB")
        else:
            logger.info(f"[{context}] No child processes")
            
        # Log system memory
        system_mem = psutil.virtual_memory()
        logger.info(f"[{context}] System memory: {system_mem.percent:.1f}% used ({system_mem.available / (1024**3):.1f} GB available)")
        
    except Exception as e:
        logger.warning(f"Could not log memory usage: {e}")

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global EMERGENCY_SHUTDOWN
    logger.warning(f"Received signal {signum}, initiating emergency shutdown...")
    EMERGENCY_SHUTDOWN = True
    
    # Try to kill child processes
    try:
        import os
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            logger.info(f"Terminating child process {child.pid}")
            child.terminate()
        
        # Wait a moment for graceful termination
        time.sleep(1)
        
        # Force kill any remaining children
        children = current_process.children(recursive=True)
        for child in children:
            logger.info(f"Force killing child process {child.pid}")
            child.kill()
            
    except Exception as e:
        logger.warning(f"Error terminating child processes: {e}")
    
    sys.exit(1)

def check_memory_usage():
    """Check if memory usage is too high."""
    try:
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > MAX_MEMORY_PERCENT:
            logger.error(f"Memory usage too high: {memory_percent:.1f}% > {MAX_MEMORY_PERCENT}%")
            return True
        return False
    except Exception as e:
        logger.warning(f"Could not check memory usage: {e}")
        return False

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

def process_shard_safely(games: List[Tuple[str, str]], shard_idx: int, processed_path: Path) -> Optional[Path]:
    """Process a shard of games with memory monitoring."""
    global EMERGENCY_SHUTDOWN
    
    if EMERGENCY_SHUTDOWN:
        return None
    
    logger.info(f"Processing shard {shard_idx:04d} with {len(games)} games")
    log_memory_usage(f"Before shard {shard_idx:04d}")
    
    # Process games in parallel with reduced workers
    processed_games = []
    games_processed = 0
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all games for processing
        future_to_game = {executor.submit(process_single_game, game): game for game in games}
        
        # Collect results with progress bar and memory monitoring
        for future in tqdm(as_completed(future_to_game), total=len(games), 
                          desc=f"Processing shard {shard_idx:04d}"):
            
            if EMERGENCY_SHUTDOWN:
                logger.warning("Emergency shutdown requested, stopping shard processing")
                return None
            
            result = future.result()
            if result is not None:
                processed_games.append(result)
            
            games_processed += 1
            
            # Check memory every MEMORY_CHECK_INTERVAL games
            if games_processed % MEMORY_CHECK_INTERVAL == 0:
                if check_memory_usage():
                    logger.error("Memory usage too high, initiating emergency shutdown")
                    EMERGENCY_SHUTDOWN = True
                    return None
    
    if not processed_games:
        logger.warning(f"No valid games in shard {shard_idx:04d}")
        return None
    
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
    
    # Save shard
    shard_file = processed_path / f"shard_{shard_idx:05d}.pkl.gz"
    with gzip.open(shard_file, 'wb') as f:
        pickle.dump(shard_data, f)
    
    logger.info(f"Created shard {shard_file} with {len(processed_games)} games")
    
    # Explicitly free memory
    logger.info(f"Freeing memory for shard {shard_idx:04d}")
    del processed_games, boards, policies, values, shard_data
    gc.collect()
    log_memory_usage(f"After shard {shard_idx:04d} cleanup")
    
    return shard_file

def process_all_data_safely(
    source_dir: str = "data/twoNetGames",
    processed_dir: str = "data/processed",
    games_per_shard: int = 1000,
    compress: bool = True
) -> List[Path]:
    """Process all data safely with memory monitoring and emergency shutdown."""
    global EMERGENCY_SHUTDOWN
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    source_path = Path(source_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    # Find all .trmph files
    trmph_files = list(source_path.glob("*.trmph"))
    trmph_files.sort()  # Ensure consistent ordering

    if not trmph_files:
        raise FileNotFoundError(f"No .trmph files found in {source_dir}")

    logger.info(f"Found {len(trmph_files)} files to process")
    logger.info(f"Using {MAX_WORKERS} workers for parallel processing")
    logger.info(f"Memory limit: {MAX_MEMORY_PERCENT}%")
    logger.info("Processing files until we have enough games for each shard")

    start_time = time.time()
    all_shard_files = []
    global_game_buffer = []
    global_shard_idx = 0
    total_games = 0
    files_processed = 0

    log_memory_usage("Start of processing")

    # Process files one at a time until we have enough games for a shard
    for trmph_file in trmph_files:
        if EMERGENCY_SHUTDOWN:
            logger.error("Emergency shutdown requested, stopping file processing")
            break
        
        files_processed += 1
        logger.info(f"Loading games from file {files_processed}/{len(trmph_files)}: {trmph_file.name}")
        
        # Check memory before loading new file
        if check_memory_usage():
            logger.error("Memory usage too high before loading file, initiating emergency shutdown")
            EMERGENCY_SHUTDOWN = True
            break
        
        games = load_games_from_file_fast(trmph_file)
        if not games:
            logger.warning(f"No games found in {trmph_file}")
            continue
        
        global_game_buffer.extend(games)
        total_games += len(games)
        
        logger.info(f"Buffer now contains {len(global_game_buffer)} games")
        log_memory_usage(f"After loading file {files_processed}")

        # Process shards while we have enough games
        while len(global_game_buffer) >= games_per_shard and not EMERGENCY_SHUTDOWN:
            shard_games = global_game_buffer[:games_per_shard]
            global_game_buffer = global_game_buffer[games_per_shard:]

            # Process this shard
            shard_file = process_shard_safely(shard_games, global_shard_idx, processed_path)
            
            if shard_file is None:
                logger.error("Shard processing failed, stopping")
                EMERGENCY_SHUTDOWN = True
                break
            
            all_shard_files.append(shard_file)
            global_shard_idx += 1
            
            # Log memory after each shard
            log_memory_usage(f"After shard {global_shard_idx-1:04d} complete")

    # Process any remaining games (final shard)
    if global_game_buffer and not EMERGENCY_SHUTDOWN:
        logger.info(f"Processing final shard with {len(global_game_buffer)} games")
        shard_file = process_shard_safely(global_game_buffer, global_shard_idx, processed_path)
        if shard_file is not None:
            all_shard_files.append(shard_file)

    processing_time = time.time() - start_time
    
    if EMERGENCY_SHUTDOWN:
        logger.error("Processing stopped due to emergency shutdown")
        logger.error("You may need to manually clean up any orphaned processes")
        return all_shard_files
    
    final_shards = list(processed_path.glob("*.pkl.gz"))
    logger.info(f"\n{'='*60}")
    logger.info("SAFE SHARD PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Files processed: {files_processed}/{len(trmph_files)}")
    logger.info(f"Total games processed: {total_games}")
    logger.info(f"Shards created: {len(all_shard_files)}")
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
    
    # Process all data safely
    try:
        process_all_data_safely()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise 