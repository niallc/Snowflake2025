"""
Enhanced data processing for Hex AI with metadata and flexible sampling.

This module implements the enhanced trmph → pkl.gz processing pipeline with:
- Comprehensive metadata tracking
- Flexible value sampling tiers
- Memory-efficient stratified processing
- Game correlation breaking
- Repeated moves handling
"""

import torch
import numpy as np
import gzip
import pickle
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import time

from .data_utils import (
    strip_trmph_preamble, split_trmph_moves, create_board_from_moves,
    create_policy_target, BOARD_SIZE, POLICY_OUTPUT_SIZE
)
from .utils.format_conversion import parse_trmph_game_record

# See setup_logging in training_utils.py for logging configuration
logger = logging.getLogger(__name__)


def assign_value_sample_tiers(total_positions: int) -> List[int]:
    """
    Assign sampling tiers to positions in a game.
    
    Tiers:
    - 0: High priority (5 positions) - Always used for value training
    - 1: Medium priority (5 positions) - Usually used for value training  
    - 2: Low priority (10 positions) - Sometimes used for value training
    - 3: Very low priority (20+ positions) - Rarely used for value training
    
    This allows flexible control over how many positions per game are used
    for value network training while keeping all positions for policy training.
    """
    if total_positions <= 5:
        # Small games: all positions get tier 0
        return [0] * total_positions
    
    # Define positions per tier
    positions_per_tier = [5, 5, 10, max(0, total_positions - 20)]
    
    # Assign tiers
    tiers = []
    for tier, count in enumerate(positions_per_tier):
        if count > 0:
            tiers.extend([tier] * min(count, total_positions - len(tiers)))
    
    # Shuffle within each tier to avoid bias
    tier_groups = {}
    for i, tier in enumerate(tiers):
        if tier not in tier_groups:
            tier_groups[tier] = []
        tier_groups[tier].append(i)
    
    # Shuffle each tier group
    for tier in tier_groups:
        random.shuffle(tier_groups[tier])
    
    # Reconstruct tiers list
    result = [0] * total_positions
    for tier, indices in tier_groups.items():
        for idx in indices:
            result[idx] = tier
    
    return result


def remove_repeated_moves(moves: List[str]) -> List[str]:
    """
    Remove repeated moves and all subsequent moves from the game.
    
    Args:
        moves: List of TRMPH moves
        
    Returns:
        Cleaned list of moves with no repetitions
    """
    seen_moves = set()
    clean_moves = []
    
    for move in moves:
        if move in seen_moves:
            # Found repeated move - discard this and all subsequent moves
            logger.debug(f"Repeated move {move} found, discarding game from this point")
            break
        seen_moves.add(move)
        clean_moves.append(move)
    
    return clean_moves


def extract_training_examples_from_game_v2(
    trmph_text: str, 
    winner_from_file: str = None,
    game_id: Tuple[int, int] = None,  # (file_idx, line_idx)
    include_trmph: bool = False,       # Whether to include full TRMPH string
    shuffle_positions: bool = True
) -> List[Dict]:
    """
    Extract training examples with comprehensive metadata and flexible sampling.
    
    Args:
        trmph_text: Complete TRMPH string
        winner_from_file: Winner from file data ("1" for blue, "2" for red)
        game_id: Tuple of (file_index, line_index) for tracking
        include_trmph: Whether to include full TRMPH string in metadata
        shuffle_positions: Whether to shuffle position order within game
        
    Returns:
        List of enhanced training examples with metadata
    """
    try:
        # Parse moves and validate
        bare_moves = strip_trmph_preamble(trmph_text)
        moves = split_trmph_moves(bare_moves)
        
        # Handle repeated moves
        moves = remove_repeated_moves(moves)
        
        if not moves:
            raise ValueError("Empty game after removing repeated moves")
        
        # Validate winner and convert to clear format
        if winner_from_file not in ["1", "2"]:
            raise ValueError(f"Invalid winner format: {winner_from_file}")
        
        # Convert winner format: "1"=BLUE, "2"=RED
        winner_clear = "BLUE" if winner_from_file == "1" else "RED"
        value_target = 0.0 if winner_from_file == "1" else 1.0  # BLUE=0.0, RED=1.0
        
        total_positions = len(moves) + 1
        
        # Assign sampling tiers
        value_sample_tiers = assign_value_sample_tiers(total_positions)
        
        # Create position indices (shuffle if requested)
        position_indices = list(range(total_positions))
        if shuffle_positions:
            random.shuffle(position_indices)
        
        training_examples = []
        
        for i, position in enumerate(position_indices):
            # Create board state
            board_state = create_board_from_moves(moves[:position])
            
            # Create policy target
            policy_target = None if position >= len(moves) else create_policy_target(moves[position])
            
            # Create metadata
            metadata = {
                'game_id': game_id,
                'position_in_game': position,
                'total_positions': total_positions,
                'value_sample_tier': value_sample_tiers[i],
                'winner': winner_clear  # Store as "BLUE" or "RED"
            }
            
            if include_trmph:
                metadata['trmph_game'] = trmph_text
            
            # Create example
            example = {
                'board': board_state,
                'policy': policy_target,
                'value': value_target,
                'metadata': metadata
            }
            
            training_examples.append(example)
        
        return training_examples
        
    except Exception as e:
        logger.error(f"Failed to extract training examples from game {trmph_text[:50]}...: {e}")
        raise ValueError(f"Failed to process game: {e}")


def extract_positions_range(
    trmph_text: str, 
    winner: str, 
    start_pos: int, 
    end_pos: int, 
    game_id: Tuple[int, int]
) -> Tuple[List[Dict], bool]:
    """
    Extract only positions in the specified range from a game.
    
    Args:
        trmph_text: Complete TRMPH string
        winner: Winner from file data ("1" or "2")
        start_pos: Starting position (inclusive)
        end_pos: Ending position (exclusive)
        game_id: Tuple of (file_index, line_index)
        
    Returns:
        List of training examples for the specified position range
    """
    try:
        # Parse moves
        bare_moves = strip_trmph_preamble(trmph_text)
        raw_moves = split_trmph_moves(bare_moves)        
        moves = remove_repeated_moves(raw_moves)
        repeat = False
        if len(raw_moves) != len(moves):
            repeat = True
        
        if not moves:
            return [], False
        
        # Validate winner
        if winner not in ["1", "2"]:
            raise ValueError(f"Invalid winner format: {winner}")
        
        winner_clear = "BLUE" if winner == "1" else "RED"
        value_target = 0.0 if winner == "1" else 1.0
        
        total_positions = len(moves) + 1
        examples = []
        
        # Extract positions in range
        for position in range(start_pos, min(end_pos, total_positions)):
            # Create board state
            board_state = create_board_from_moves(moves[:position])
            
            # Create policy target
            policy_target = None if position >= len(moves) else create_policy_target(moves[position])
            
            # Create metadata
            metadata = {
                'game_id': game_id,
                'position_in_game': position,
                'total_positions': total_positions,
                'value_sample_tier': 0,  # Default tier for range extraction
                'winner': winner_clear
            }
            
            # Create example
            example = {
                'board': board_state,
                'policy': policy_target,
                'value': value_target,
                'metadata': metadata
            }
            
            examples.append(example)
        
        return examples, repeat
        
    except Exception as e:
        logger.error(f"Failed to extract positions range from game: {e}")
        return [], False


def create_file_lookup_table(trmph_files: List[Path], output_dir: Path) -> Path:
    """
    Create a file lookup table mapping file indices to actual filenames.
    
    Args:
        trmph_files: List of TRMPH file paths
        output_dir: Directory to save the lookup table
        
    Returns:
        Path to the created lookup table file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lookup_file = output_dir / f"file_lookup_{timestamp}.json"
    
    file_mapping = {}
    for file_idx, file_path in enumerate(trmph_files):
        file_mapping[file_idx] = str(file_path)
    
    lookup_data = {
        'file_mapping': file_mapping,
        'created_at': datetime.now().isoformat(),
        'total_files': len(trmph_files),
        'format_version': '1.0'
    }
    
    with open(lookup_file, 'w') as f:
        json.dump(lookup_data, f, indent=2)
    
    logger.info(f"Created file lookup table: {lookup_file}")
    return lookup_file


def create_stratified_dataset(
    trmph_files: List[Path], 
    output_dir: Path, 
    positions_per_pass: int = 5,
    include_trmph: bool = False,
    max_positions_per_game: int = 169 # 13x13 board
) -> Tuple[List[Path], Path]:
    """
    Create dataset in multiple passes to break game correlations while managing memory.
    
    Pass 1: Process positions 0-4 from all games
    Pass 2: Process positions 5-9 from all games  
    Pass 3: Process positions 10-14 from all games
    etc.
    
    This creates a 'striped' dataset where each file contains positions from
    different stages of games, breaking temporal correlations.
    
    Args:
        trmph_files: List of TRMPH file paths
        output_dir: Directory to save processed files
        positions_per_pass: Number of positions to process per pass
        include_trmph: Whether to include full TRMPH strings in metadata
        max_positions_per_game: Maximum positions to consider per game
        
    Returns:
        Tuple of (processed_file_paths, lookup_table_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create file lookup table
    lookup_file = create_file_lookup_table(trmph_files, output_dir)
    
    processed_files = []
    total_examples = 0
    
    # Determine number of passes needed
    max_positions = max_positions_per_game
    
    logger.info(f"Starting stratified processing with {positions_per_pass} positions per pass")
    logger.info(f"Maximum positions per game: {max_positions}")
    
    for pass_num in range(0, max_positions, positions_per_pass):
        start_pos = pass_num
        end_pos = min(pass_num + positions_per_pass, max_positions)
        
        logger.info(f"Processing pass {pass_num//positions_per_pass + 1}: positions {start_pos}-{end_pos}")
        
        # Process all games for this position range
        examples = []
        games_processed = 0
        repeat_games = 0
        
        for file_idx, trmph_file in enumerate(trmph_files):
            try:
                # Load games from file
                from .data_utils import load_trmph_file
                games = load_trmph_file(str(trmph_file))
                
                for line_idx, game_line in enumerate(games):
                    try:
                        trmph_url, winner = parse_trmph_game_record(game_line)
                        game_id = (file_idx, line_idx)
                        
                        game_examples, repeat_game = extract_positions_range(
                            trmph_url, winner, start_pos, end_pos, game_id
                        )
                        
                        if game_examples:
                            examples.extend(game_examples)
                            games_processed += 1
                            if repeat_game:
                                repeat_games += 1
                            
                    except Exception as e:
                        logger.warning(f"Error processing game {line_idx} in {trmph_file.name}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error loading file {trmph_file}: {e}")
                continue
        
        if examples:
            # Shuffle within this pass
            random.shuffle(examples)
            
            # Save pass-specific file
            output_file = output_dir / f"pass_{pass_num:03d}_positions_{start_pos}-{end_pos}.pkl.gz"
            
            with gzip.open(output_file, 'wb') as f:
                pickle.dump({
                    'examples': examples,
                    'pass_info': {
                        'pass_num': pass_num,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'games_processed': games_processed,
                        'examples_count': len(examples)
                    },
                    'processed_at': datetime.now().isoformat(),
                    'format_version': '2.0'
                }, f)
            
            processed_files.append(output_file)
            total_examples += len(examples)
            
            logger.info(f"Saved pass {pass_num//positions_per_pass + 1}: "
                       f"{len(examples)} examples from {games_processed} games")
        else:
            logger.warning(f"No examples generated for pass {pass_num//positions_per_pass + 1}")
    
    logger.info(f"Stratified processing complete: {len(processed_files)} files, {total_examples} total examples")
    logger.info(f"Number of repeat games: {repeat_games}")
    return processed_files, lookup_file


def create_chunked_dataset(
    trmph_files: List[Path], 
    output_dir: Path,
    games_per_chunk: int = 10000,
    include_trmph: bool = False
) -> Tuple[List[Path], Path]:
    """
    Process games in chunks, but ensure chunks don't align with file boundaries.
    
    Args:
        trmph_files: List of TRMPH file paths
        output_dir: Directory to save processed files
        games_per_chunk: Number of games per chunk
        include_trmph: Whether to include full TRMPH strings in metadata
        
    Returns:
        Tuple of (processed_file_paths, lookup_table_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create file lookup table
    lookup_file = create_file_lookup_table(trmph_files, output_dir)
    
    # Collect all games with their metadata
    logger.info("Collecting all games...")
    all_games = []
    
    for file_idx, trmph_file in enumerate(trmph_files):
        try:
            from .data_utils import load_trmph_file
            games = load_trmph_file(str(trmph_file))
            
            for line_idx, game_line in enumerate(games):
                all_games.append((file_idx, line_idx, game_line))
                
        except Exception as e:
            logger.error(f"Error loading file {trmph_file}: {e}")
            continue
    
    logger.info(f"Collected {len(all_games)} games")
    
    # Shuffle game order
    random.shuffle(all_games)
    
    processed_files = []
    total_examples = 0
    
    # Process in chunks
    for chunk_idx in range(0, len(all_games), games_per_chunk):
        chunk_games = all_games[chunk_idx:chunk_idx + games_per_chunk]
        examples = []
        
        logger.info(f"Processing chunk {chunk_idx//games_per_chunk + 1}: {len(chunk_games)} games")
        
        for file_idx, line_idx, game_line in chunk_games:
            try:
                trmph_url, winner = parse_trmph_game_record(game_line)
                game_id = (file_idx, line_idx)
                
                game_examples = extract_training_examples_from_game_v2(
                    trmph_url, winner, game_id, include_trmph, shuffle_positions=True
                )
                examples.extend(game_examples)
                
            except Exception as e:
                logger.warning(f"Error processing game: {e}")
                continue
        
        if examples:
            # Shuffle within chunk
            random.shuffle(examples)
            
            # Save chunk
            output_file = output_dir / f"chunk_{chunk_idx:06d}.pkl.gz"
            
            with gzip.open(output_file, 'wb') as f:
                pickle.dump({
                    'examples': examples,
                    'chunk_info': {
                        'chunk_idx': chunk_idx,
                        'games_processed': len(chunk_games),
                        'examples_count': len(examples)
                    },
                    'processed_at': datetime.now().isoformat(),
                    'format_version': '2.0'
                }, f)
            
            processed_files.append(output_file)
            total_examples += len(examples)
            
            logger.info(f"Saved chunk {chunk_idx//games_per_chunk + 1}: {len(examples)} examples")
    
    logger.info(f"Chunked processing complete: {len(processed_files)} files, {total_examples} total examples")
    return processed_files, lookup_file 