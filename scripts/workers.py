"""
Worker functions for multiprocessing TRMPH file processing.

This module contains worker functions that can be safely pickled and used
with multiprocessing. All functions must be at module level (not nested)
to avoid pickling issues.
"""

import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Import hex_ai modules
from hex_ai.batch_processor import BatchProcessor
from hex_ai.file_utils import atomic_write_pickle_gz, sanitize_filename
from hex_ai.data_utils import load_trmph_file
from hex_ai.utils.format_conversion import parse_trmph_game_record
from hex_ai.data_utils import extract_training_examples_with_selector_from_game

logger = logging.getLogger(__name__)


def process_single_file_worker(file_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for processing a single TRMPH file.
    
    This function must be at module level (not nested) to be picklable.
    It receives all necessary information as a single dictionary to avoid
    complex argument passing.
    
    Args:
        file_info: Dict containing:
            - file_path: str - Path to TRMPH file
            - file_idx: int - Index of file in processing list
            - data_dir: str - Data directory path
            - output_dir: str - Output directory path
            - run_tag: str - Optional run tag for output files
            - position_selector: str - Position selector for extraction
    
    Returns:
        Dict with processing results:
            - success: bool - Whether processing succeeded
            - stats: dict - Processing statistics (if successful)
            - error: str - Error message (if failed)
            - file_path: str - Original file path
            - file_idx: int - File index
    """
    try:
        # Unpack configuration
        file_path = Path(file_info['file_path'])
        file_idx = file_info['file_idx']
        data_dir = file_info['data_dir']
        output_dir = Path(file_info['output_dir'])
        run_tag = file_info.get('run_tag')
        position_selector = file_info.get('position_selector', 'all')
        
        # Process the file directly without BatchProcessor to avoid resume issues
        logger.info(f"Processing file {file_path} (index {file_idx})")
        stats = process_single_file_direct(
            file_path, 
            file_idx, 
            output_dir,
            position_selector=position_selector
        )
        logger.info(f"Completed processing {file_path}: {stats.get('examples_generated', 0)} examples")
        
        return {
            'success': True,
            'stats': stats,
            'file_path': str(file_path),
            'file_idx': file_idx
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file_path': file_info.get('file_path', 'unknown'),
            'file_idx': file_info.get('file_idx', -1)
        }


def process_single_file_direct(file_path: Path, file_idx: int, output_dir: Path, position_selector: str = "all") -> Dict[str, Any]:
    """
    Process a single .trmph file directly without BatchProcessor state management.
    
    This function avoids the resume logic that causes race conditions in multiprocessing.
    """
    file_stats = {
        'file_path': str(file_path),
        'all_games': 0,           # Total games attempted (including invalid ones)
        'valid_games': 0,         # Successfully processed games
        'skipped_games': 0,       # Games that couldn't be processed (format errors, etc.)
        'examples_generated': 0,  # Total training examples created
        'file_error': None        # File-level error (if any)
    }
    
    try:
        logger.info(f"Processing {file_path}")
        
        # Load the trmph file
        try:
            games = load_trmph_file(str(file_path))
            logger.info(f"  Loaded {len(games)} games from {file_path}")
        except (FileNotFoundError, ValueError) as e:
            # File-level error - can't process this file at all
            file_stats['file_error'] = str(e)
            logger.error(f"File error processing {file_path}: {e}")
            return file_stats
        
        # Process each game
        all_examples = []
        for i, game_line in enumerate(games):
            file_stats['all_games'] += 1
            
            try:
                # Parse the game record
                try:
                    trmph_url, winner = parse_trmph_game_record(game_line)
                except ValueError as e:
                    logger.warning(f"    Game {i+1} has wrong format: {repr(game_line)}: {e}")
                    file_stats['skipped_games'] += 1
                    continue
                
                # Extract training examples
                try:
                    # Create game_id with file_idx and line_idx (i+1 for 1-based line numbers)
                    game_id = (file_idx, i+1)
                    examples = extract_training_examples_with_selector_from_game(trmph_url, winner, game_id, position_selector=position_selector)
                    if examples:
                        # Use dictionary format directly - no conversion needed
                        all_examples.extend(examples)
                        file_stats['valid_games'] += 1
                        file_stats['examples_generated'] += len(examples)
                    else:
                        logger.warning(f"    Game {i+1} in {file_path.name} produced no examples")
                        file_stats['skipped_games'] += 1
                except Exception as e:
                    logger.warning(f"    Error extracting examples from game {i+1} in {file_path.name}: {e}")
                    file_stats['skipped_games'] += 1
            
            except Exception as e:
                logger.warning(f"    Unexpected error processing game {i+1} in {file_path.name}: {e}")
                file_stats['skipped_games'] += 1
        
        # Save processed examples
        if all_examples:
            # Sanitize filename and ensure uniqueness
            safe_filename = sanitize_filename(file_path.stem)
            output_file = output_dir / f"{safe_filename}_processed.pkl.gz"
            
            # Ensure filename uniqueness
            counter = 1
            while output_file.exists():
                output_file = output_dir / f"{safe_filename}_processed_{counter}.pkl.gz"
                counter += 1
                if counter > 4:  # Prevent infinite loop
                    raise ValueError(f"Too many files with similar name: {safe_filename}")
            
            # Validate data before saving
            validate_examples_data(all_examples)
            
            try:
                # Save with atomic write
                data = {
                    'examples': all_examples,
                    'source_file': str(file_path),
                    'processing_stats': file_stats,
                    'processed_at': datetime.now().isoformat(),
                    'file_size_bytes': 0  # Will be updated after write
                }
                
                atomic_write_pickle_gz(data, output_file)
                
                # Get file size for logging
                file_size = output_file.stat().st_size
                logger.info(f"  Saved {len(all_examples)} examples to {output_file} ({file_size} bytes)")
                
            except Exception as e:
                logger.error(f"    Error saving output file {output_file}: {e}")
                file_stats['file_error'] = f"Failed to save output: {e}"
        else:
            logger.info(f"  No valid examples generated from {file_path}")
        
        return file_stats
        
    except Exception as e:
        # Catch any other unexpected file-level errors
        file_stats['file_error'] = str(e)
        logger.error(f"Unexpected error processing {file_path}: {e}")
        return file_stats


def validate_examples_data(examples: list):
    """Validate that examples have the correct format."""
    if not examples:
        return
    
    # Check first example for required fields
    example = examples[0]
    required_fields = ['board', 'policy', 'value', 'player_to_move', 'metadata']
    
    for field in required_fields:
        if field not in example:
            raise ValueError(f"Example missing required field: {field}")
    
    # Validate board shape
    board = example['board']
    if not hasattr(board, 'shape') or len(board.shape) != 3 or board.shape[0] != 2:
        raise ValueError(f"Board must be (2, N, N) array, got shape: {board.shape}")
    
    # Validate policy shape
    policy = example['policy']
    if policy is not None:
        board_size = board.shape[1] * board.shape[2]
        if not hasattr(policy, 'shape') or policy.shape[0] != board_size:
            raise ValueError(f"Policy must be ({board_size},) array, got shape: {policy.shape}")
    
    # Validate value
    value = example['value']
    if not isinstance(value, (int, float)):
        raise ValueError(f"Value must be numeric, got: {type(value)}")
    
    # Validate player_to_move
    player_to_move = example['player_to_move']
    if isinstance(player_to_move, int):
        if player_to_move not in [0, 1]:
            raise ValueError(f"player_to_move must be 0 or 1, got: {player_to_move}")
    else:
        # Check if it's a Player enum
        try:
            from hex_ai.value_utils import Player
            if player_to_move not in [Player.BLUE, Player.RED]:
                raise ValueError(f"player_to_move must be Player.BLUE or Player.RED, got: {player_to_move}")
        except ImportError:
            raise ValueError(f"player_to_move must be 0, 1, or Player enum, got: {player_to_move}")
    
    # Validate metadata
    metadata = example['metadata']
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be dict, got: {type(metadata)}")
    
    required_metadata = ['game_id', 'position_in_game', 'winner']
    for field in required_metadata:
        if field not in metadata:
            raise ValueError(f"Metadata missing required field: {field}") 