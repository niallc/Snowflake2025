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
from hex_ai.value_utils import Player, Winner

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
        
        # Load TRMPH file
        trmph_lines = load_trmph_file(file_path)
        
        # Process each game line
        all_examples = []
        for i, game_line in enumerate(trmph_lines):
            try:
                # Parse the game record
                trmph_url, winner = parse_trmph_game_record(game_line)
                
                # Extract training examples from this game
                game_id = (file_idx, i+1)  # file_idx and line_idx (1-based)
                examples = extract_training_examples_with_selector_from_game(
                    trmph_url, winner, game_id, position_selector=position_selector
                )
                
                if examples:
                    # Add source file information to each example
                    for example in examples:
                        example['metadata']['source_file'] = str(file_path)
                        example['metadata']['game_id'] = (file_idx, i+1)  # Use tuple format
                    
                    all_examples.extend(examples)
                    file_stats['valid_games'] += 1
                    file_stats['examples_generated'] += len(examples)
                else:
                    file_stats['skipped_games'] += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process game {i+1} in {file_path}: {e}")
                file_stats['skipped_games'] += 1
        
        file_stats['all_games'] = len(trmph_lines)
        
        # Save processed examples if any were generated
        if all_examples:
            # Validate examples before saving
            validate_examples_data(all_examples)
            
            # Generate output filename
            base_name = file_path.stem
            output_filename = f"{base_name}_processed.pkl.gz"
            output_path = output_dir / output_filename
            
            # Ensure filename uniqueness
            counter = 1
            while output_path.exists():
                output_filename = f"{base_name}_processed_{counter}.pkl.gz"
                output_path = output_dir / output_filename
                counter += 1
            
            # Save with atomic write
            data = {
                'examples': all_examples,
                'source_file': str(file_path),
                'processing_stats': file_stats,
                'processed_at': datetime.now().isoformat()
            }
            atomic_write_pickle_gz(data, output_path)
            
            logger.info(f"Saved {len(all_examples)} examples to {output_path}")
        
        return file_stats
        
    except Exception as e:
        file_stats['file_error'] = str(e)
        logger.error(f"Failed to process file {file_path}: {e}")
        raise


def validate_examples_data(examples: list):
    """
    Validate that examples have the correct format and required fields.
    
    Args:
        examples: List of training examples to validate
        
    Raises:
        ValueError: If examples don't have required format
    """
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
    if not hasattr(board, 'shape') or len(board.shape) != 3:
        raise ValueError(f"Board must be 3D array, got shape: {board.shape}")
    
    if board.shape[0] != 2:
        raise ValueError(f"Board must have 2 channels, got: {board.shape[0]}")
    
    # Validate policy shape
    policy = example['policy']
    if policy is not None:
        board_size = board.shape[1] * board.shape[2]
        if not hasattr(policy, 'shape') or policy.shape[0] != board_size:
            raise ValueError(f"Policy must have shape ({board_size},), got: {policy.shape}")
    
    # Validate player_to_move (should be Player enum)
    player_to_move = example['player_to_move']
    if not isinstance(player_to_move, Player):
        raise ValueError(f"player_to_move must be Player enum, got: {type(player_to_move)}: {player_to_move}")
    
    # Validate metadata
    metadata = example['metadata']
    required_metadata = ['game_id', 'position_in_game', 'winner']
    for field in required_metadata:
        if field not in metadata:
            raise ValueError(f"Metadata missing required field: {field}")
    
    # Validate winner values (should be Winner enum or None)
    winner = metadata['winner']
    if winner is not None and not isinstance(winner, Winner):
        raise ValueError(f"Winner must be Winner enum or None, got: {type(winner)}: {winner}") 