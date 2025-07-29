import torch
import numpy as np
import time
import psutil
from typing import Union, Tuple, List, Optional, Dict, Any
from collections import OrderedDict
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.utils import format_conversion as fc
from hex_ai.inference.board_display import display_hex_board
from hex_ai.config import (
    PLAYER_CHANNEL, MODEL_CHANNELS,
    BLUE_PLAYER, RED_PLAYER, BLUE_CHANNEL, RED_CHANNEL,
    BLUE_PIECE, RED_PIECE, EMPTY_PIECE,
    TRAINING_BLUE_WIN, TRAINING_RED_WIN,
    TRMPH_BLUE_WIN, TRMPH_RED_WIN,
)
from hex_ai.data_utils import create_board_from_moves, preprocess_example_for_model, get_player_to_move_from_board
from hex_ai.value_utils import trmph_winner_to_training_value, trmph_winner_to_clear_str, model_output_to_prob, ValuePerspective
from hex_ai.value_utils import (
    # Add new utilities
    policy_logits_to_probs,
    get_legal_policy_probs,
    select_top_k_moves,
    get_top_k_moves_with_probs,
)


class LRUCache:
    """Simple LRU cache for board position caching."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Tuple[np.ndarray, float]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key: str, value: Tuple[np.ndarray, float]):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        self.cache[key] = value
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0,
            'size': len(self.cache),
            'max_size': self.max_size
        }


class SimpleModelInference:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        model_type: str = "resnet18",
        cache_size: int = 10000,
        max_batch_size: int = 1000,
        enable_caching: bool = True
    ):
        print(f"SimpleModelInference.__init__() called with checkpoint_path={checkpoint_path}, device={device}, model_type={model_type}")
        self.model = ModelWrapper(checkpoint_path, device=device, model_type=model_type)
        self.board_size = fc.BOARD_SIZE
        self.checkpoint_path = checkpoint_path
        self.max_batch_size = max_batch_size
        self.enable_caching = enable_caching
        
        # Initialize cache if enabled
        if self.enable_caching:
            self.cache = LRUCache(cache_size)
        else:
            self.cache = None
        
        # Performance tracking
        self.stats = {
            'total_inferences': 0,
            'total_batch_inferences': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        print(f"Using model architecture (expects {MODEL_CHANNELS} channels)")
        if self.enable_caching:
            print(f"Cache enabled with size {cache_size}")

    def _get_board_hash(self, board: Union[str, np.ndarray, torch.Tensor]) -> str:
        """Create a hash for caching board positions."""
        if isinstance(board, str):
            return board  # Use TRMPH string as hash
        elif isinstance(board, np.ndarray):
            return str(board.tobytes())
        elif isinstance(board, torch.Tensor):
            return str(board.cpu().numpy().tobytes())
        else:
            raise TypeError(f"Unsupported board type for hashing: {type(board)}")

    def _check_cache(self, board: Union[str, np.ndarray, torch.Tensor]) -> Optional[Tuple[np.ndarray, float]]:
        """Check cache for board position."""
        if not self.enable_caching or self.cache is None:
            return None
        
        board_hash = self._get_board_hash(board)
        return self.cache.get(board_hash)

    def _store_in_cache(self, board: Union[str, np.ndarray, torch.Tensor], result: Tuple[np.ndarray, float]):
        """Store result in cache."""
        if not self.enable_caching or self.cache is None:
            return
        
        board_hash = self._get_board_hash(board)
        self.cache.put(board_hash, result)

    def _calculate_optimal_batch_size(self, total_boards: int) -> int:
        """Calculate optimal batch size based on available memory."""
        batch_size = min(self.max_batch_size, total_boards)
        
        # Check system memory
        try:
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            # Estimate memory per board (3 channels * 13 * 13 * 4 bytes)
            bytes_per_board = 3 * 13 * 13 * 4  # ~2KB per board
            memory_based_batch_size = int(available_memory_mb * 0.1 / (bytes_per_board / (1024 * 1024)))
            batch_size = min(batch_size, memory_based_batch_size)
        except Exception:
            pass  # If psutil fails, use default batch size
        
        # Check GPU memory if available
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_mb = gpu_memory / (1024 * 1024)
                gpu_batch_size = int(gpu_memory_mb * 0.8 / (bytes_per_board / (1024 * 1024)))
                batch_size = min(batch_size, gpu_batch_size)
            except Exception:
                pass  # If GPU memory check fails, use current batch size
        
        return max(1, batch_size)

    def trmph_to_2nxn(self, trmph: str) -> torch.Tensor:
        board_nxn = fc.parse_trmph_to_board(trmph, board_size=self.board_size)
        # TODO: This function is legacy and only used internally; prefer board_nxn_to_2nxn for inference
        return fc.board_nxn_to_2nxn(board_nxn)

    def _is_finished_position(self, board_2ch: np.ndarray) -> Tuple[bool, int]:
        """
        Check if a position is finished (one player has won).
        Returns (is_finished, winner) where winner is BLUE_PLAYER or RED_PLAYER.
        """
        # For now, use a simple heuristic: if one player has significantly more pieces
        # and the board has a reasonable number of pieces, it's likely a finished position
        blue_pieces = np.sum(board_2ch[BLUE_CHANNEL])  # Use BLUE_CHANNEL constant
        red_pieces = np.sum(board_2ch[RED_CHANNEL])    # Use RED_CHANNEL constant
        total_pieces = blue_pieces + red_pieces
        
        # If board has a reasonable number of pieces (>10), check for winner
        if total_pieces > 10:
            # Simple heuristic: player with more pieces likely won
            if blue_pieces > red_pieces:
                return True, BLUE_PLAYER  # Use BLUE_PLAYER constant
            elif red_pieces > blue_pieces:
                return True, RED_PLAYER   # Use RED_PLAYER constant
        
        return False, -1

    def _create_board_with_correct_player_channel(self, board_2ch: np.ndarray) -> torch.Tensor:
        """
        Create a 3-channel board tensor with the correct player-to-move channel.
        For finished positions, sets the player-to-move to the loser (next player),
        which is how the model was trained.
        """
        is_finished, winner = self._is_finished_position(board_2ch)
        
        if is_finished:
            # For finished positions, set player-to-move to the loser (next player)
            # This matches how the model was trained on final positions
            # NOTE: This is the key insight - training data has player-to-move = loser
            loser = RED_PLAYER if winner == BLUE_PLAYER else BLUE_PLAYER
            player_to_move = loser
        else:
            # For non-finished positions, use normal logic
            player_to_move = get_player_to_move_from_board(board_2ch)
            # Convert Player enum to integer for tensor creation
            player_to_move = player_to_move.value
        
        # Create player-to-move channel
        # NOTE: player_to_move is now an integer (0 for BLUE, 1 for RED)
        player_channel = np.full((1, self.board_size, self.board_size), float(player_to_move), dtype=board_2ch.dtype)
        board_3ch = np.concatenate([board_2ch, player_channel], axis=0)
        return torch.tensor(board_3ch, dtype=torch.float32)

    def simple_infer(self, board: Union[str, np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, float]:
        """
        Accepts a board in trmph string, (N,N) np.ndarray, or (2,N,N) or (3,N,N) torch.Tensor format.
        Uses the same preprocessing pipeline as training to ensure consistency.
        Returns (policy_logits, value_logit):
            - policy_logits: np.ndarray, raw policy logits (before softmax)
            - value_logit: float, raw value logit (before sigmoid)
        """
        start_time = time.time()
        
        # Check cache first
        cached_result = self._check_cache(board)
        if cached_result is not None:
            self.stats['cache_hits'] += 1
            return cached_result
        
        self.stats['cache_misses'] += 1
        
        # Convert input to the same format used in training
        if isinstance(board, str):
            # For TRMPH strings, use the same pipeline as training
            board_2ch = create_board_from_moves(fc.split_trmph_moves(fc.strip_trmph_preamble(board)))
        elif isinstance(board, np.ndarray):
            if board.shape == (self.board_size, self.board_size):
                # NxN format - convert to 2-channel using same logic as training
                # NOTE: NxN boards use BLUE_PIECE (1) and RED_PIECE (2)
                board_2ch = np.zeros((2, self.board_size, self.board_size), dtype=np.float32)
                board_2ch[BLUE_CHANNEL] = (board == BLUE_PIECE).astype(np.float32)  # Blue channel
                board_2ch[RED_CHANNEL] = (board == RED_PIECE).astype(np.float32)   # Red channel
            elif board.shape == (2, self.board_size, self.board_size):
                # Already 2-channel format
                board_2ch = board
            elif board.shape == (3, self.board_size, self.board_size):
                # 3-channel format - extract first 2 channels
                board_2ch = board[:2]
            else:
                raise ValueError(f"Numpy array must have shape ({self.board_size}, {self.board_size}), (2, {self.board_size}, {self.board_size}), or (3, {self.board_size}, {self.board_size})")
        elif isinstance(board, torch.Tensor):
            if board.shape == (2, self.board_size, self.board_size):
                # Already 2-channel format
                board_2ch = board.cpu().numpy()
            elif board.shape == (3, self.board_size, self.board_size):
                # 3-channel format - extract first 2 channels
                board_2ch = board[:2].cpu().numpy()
            else:
                raise ValueError(f"Tensor must have shape (2, {self.board_size}, {self.board_size}) or (3, {self.board_size}, {self.board_size})")
        else:
            raise TypeError("Board must be a trmph string, (N,N) np.ndarray, or (2,N,N) or (3,N,N) torch.Tensor")

        # Create the 3-channel board with correct player-to-move channel
        input_tensor = self._create_board_with_correct_player_channel(board_2ch)

        policy_logits, value_logit = self.model.predict(input_tensor)
        policy_logits_np = policy_logits.detach().cpu().numpy() if hasattr(policy_logits, 'detach') else np.array(policy_logits)
        value_logit_val = value_logit.item() if hasattr(value_logit, 'item') else float(value_logit)
        
        result = (policy_logits_np, value_logit_val)
        
        # Store in cache
        self._store_in_cache(board, result)
        
        # Update performance stats
        self.stats['total_inferences'] += 1
        self.stats['total_time'] += time.time() - start_time
        
        return result

    def infer(self, board: Union[str, np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, float]:
        """
        Alias for simple_infer for backward compatibility.
        """
        print(f"WARNING: SimpleModelInference.infer() is deprecated. Use simple_infer() instead.")
        return self.simple_infer(board)

    def batch_infer(self, boards: List[Union[str, np.ndarray, torch.Tensor]]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Efficiently process multiple boards in a single batch.
        Args:
            boards: List of boards in any supported format (trmph string, (N,N) np.ndarray, or (2,N,N) or (3,N,N) torch.Tensor)
        Returns:
            Tuple of (policy_logits_list, value_logits_list) where each list contains the results for each board
        """
        if not boards:
            return [], []
        
        start_time = time.time()
        
        # Check cache for each board
        uncached_boards = []
        uncached_indices = []
        results = [None] * len(boards)
        
        for i, board in enumerate(boards):
            cached_result = self._check_cache(board)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                results[i] = cached_result
            else:
                self.stats['cache_misses'] += 1
                uncached_boards.append(board)
                uncached_indices.append(i)
        
        # Process uncached boards in optimal batch sizes
        if uncached_boards:
            optimal_batch_size = self._calculate_optimal_batch_size(len(uncached_boards))
            
            # Convert all uncached boards to 3-channel tensors
            input_tensors = []
            for board in uncached_boards:
                # Convert input to the same format used in training
                if isinstance(board, str):
                    # For TRMPH strings, use the same pipeline as training
                    board_2ch = create_board_from_moves(fc.split_trmph_moves(fc.strip_trmph_preamble(board)))
                elif isinstance(board, np.ndarray):
                    if board.shape == (self.board_size, self.board_size):
                        # NxN format - convert to 2-channel using same logic as training
                        board_2ch = np.zeros((2, self.board_size, self.board_size), dtype=np.float32)
                        board_2ch[BLUE_CHANNEL] = (board == BLUE_PIECE).astype(np.float32)  # Blue channel
                        board_2ch[RED_CHANNEL] = (board == RED_PIECE).astype(np.float32)   # Red channel
                    elif board.shape == (2, self.board_size, self.board_size):
                        # Already 2-channel format
                        board_2ch = board
                    elif board.shape == (3, self.board_size, self.board_size):
                        # 3-channel format - extract first 2 channels
                        board_2ch = board[:2]
                    else:
                        raise ValueError(f"Numpy array must have shape ({self.board_size}, {self.board_size}), (2, {self.board_size}, {self.board_size}), or (3, {self.board_size}, {self.board_size})")
                elif isinstance(board, torch.Tensor):
                    if board.shape == (2, self.board_size, self.board_size):
                        # Already 2-channel format
                        board_2ch = board.cpu().numpy()
                    elif board.shape == (3, self.board_size, self.board_size):
                        # 3-channel format - extract first 2 channels
                        board_2ch = board[:2].cpu().numpy()
                    else:
                        raise ValueError(f"Tensor must have shape (2, {self.board_size}, {self.board_size}) or (3, {self.board_size}, {self.board_size})")
                else:
                    raise TypeError("Board must be a trmph string, (N,N) np.ndarray, or (2,N,N) or (3,N,N) torch.Tensor")

                # Create the 3-channel board with correct player-to-move channel
                input_tensor = self._create_board_with_correct_player_channel(board_2ch)
                input_tensors.append(input_tensor)
            
            # Process in batches
            all_policies = []
            all_values = []
            
            for i in range(0, len(input_tensors), optimal_batch_size):
                batch_tensors = input_tensors[i:i + optimal_batch_size]
                batch_tensor = torch.stack(batch_tensors)
                
                # Run batch inference
                policy_logits_batch, value_logits_batch = self.model.batch_predict(batch_tensor)
                
                # Convert results
                batch_policies = [logits.detach().cpu().numpy() for logits in policy_logits_batch]
                batch_values = [logit.item() for logit in value_logits_batch]
                
                all_policies.extend(batch_policies)
                all_values.extend(batch_values)
                
                # Clear GPU cache if needed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Store results and update cache
            for idx, board, policy, value in zip(uncached_indices, uncached_boards, all_policies, all_values):
                result = (policy, value)
                results[idx] = result
                self._store_in_cache(board, result)
        
        # Unzip results
        policies = [result[0] for result in results]
        values = [result[1] for result in results]
        
        # Update performance stats
        self.stats['total_batch_inferences'] += 1
        self.stats['total_inferences'] += len(boards)
        self.stats['total_time'] += time.time() - start_time
        
        return policies, values

    def get_top_k_moves(self, policy_logits: np.ndarray, k: int = 3, temperature: float = 1.0) -> List[Tuple[str, float]]:
        """
        Returns the top-k moves as (trmph_move, probability) tuples.
        Now accepts policy logits and uses centralized utilities.
        """
        # Convert logits to probabilities using centralized utility
        policy_probs = policy_logits_to_probs(policy_logits, temperature)
        topk_indices = np.argsort(policy_probs)[::-1][:k]
        moves = [(fc.tensor_to_trmph(idx, self.board_size), float(policy_probs[idx])) for idx in topk_indices]
        return moves

    def get_top_k_legal_moves(self, policy_logits: np.ndarray, legal_moves: List[Tuple[int, int]], k: int = 3, temperature: float = 1.0) -> List[Tuple[Tuple[int, int], float]]:
        """
        Returns the top-k legal moves as ((row, col), probability) tuples.
        Uses centralized utilities for policy processing.
        """
        return get_top_k_moves_with_probs(policy_logits, legal_moves, self.board_size, k, temperature)

    def display_board(self, board: Union[str, np.ndarray, torch.Tensor]):
        """
        Display the board using display_hex_board.
        Accepts trmph string, (N,N) np.ndarray, or (2,N,N) torch.Tensor.
        """
        if isinstance(board, str):
            board_nxn = fc.parse_trmph_to_board(board, board_size=self.board_size)
        elif isinstance(board, np.ndarray):
            board_nxn = board
        elif isinstance(board, torch.Tensor):
            board_nxn = fc.board_2nxn_to_nxn(board)
        else:
            raise TypeError("Board must be a trmph string, (N,N) np.ndarray, or (2,N,N) torch.Tensor")
        display_hex_board(board_nxn)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.stats.copy()
        
        # Add cache stats if caching is enabled
        if self.cache is not None:
            cache_stats = self.cache.get_stats()
            stats['cache'] = cache_stats
        
        # Add memory usage stats
        try:
            stats['memory'] = {
                'system_memory_mb': psutil.virtual_memory().used / (1024 * 1024),
                'system_memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
            }
            
            if torch.cuda.is_available():
                stats['memory']['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                stats['memory']['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        except Exception:
            stats['memory'] = {'error': 'Could not retrieve memory stats'}
        
        # Calculate throughput
        if stats['total_time'] > 0:
            stats['throughput'] = stats['total_inferences'] / stats['total_time']
            if stats['total_batch_inferences'] > 0:
                stats['avg_batch_size'] = stats['total_inferences'] / stats['total_batch_inferences']
        else:
            stats['throughput'] = 0.0
            stats['avg_batch_size'] = 0.0
        
        return stats

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        memory_stats = {}
        
        try:
            memory_stats['system_used_mb'] = psutil.virtual_memory().used / (1024 * 1024)
            memory_stats['system_available_mb'] = psutil.virtual_memory().available / (1024 * 1024)
            memory_stats['system_total_mb'] = psutil.virtual_memory().total / (1024 * 1024)
        except Exception:
            memory_stats['system_error'] = 'Could not retrieve system memory stats'
        
        if torch.cuda.is_available():
            try:
                memory_stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
                memory_stats['gpu_total_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            except Exception:
                memory_stats['gpu_error'] = 'Could not retrieve GPU memory stats'
        
        return memory_stats

    def clear_cache(self):
        """Clear the inference cache."""
        if self.cache is not None:
            self.cache.cache.clear()
            self.cache.hits = 0
            self.cache.misses = 0
            print("Cache cleared")

    def print_performance_summary(self):
        """Print a summary of performance statistics."""
        stats = self.get_performance_stats()
        
        print("\n=== Inference Performance Summary ===")
        print(f"Total inferences: {stats['total_inferences']}")
        print(f"Total batch inferences: {stats['total_batch_inferences']}")
        print(f"Total time: {stats['total_time']:.2f}s")
        print(f"Throughput: {stats['throughput']:.1f} boards/s")
        
        if stats['total_batch_inferences'] > 0:
            print(f"Average batch size: {stats['avg_batch_size']:.1f}")
        
        if 'cache' in stats:
            cache = stats['cache']
            print(f"Cache hits: {cache['hits']}, misses: {cache['misses']}")
            print(f"Cache hit rate: {cache['hit_rate']:.1%}")
            print(f"Cache size: {cache['size']}/{cache['max_size']}")
        
        if 'memory' in stats and 'error' not in stats['memory']:
            mem = stats['memory']
            print(f"System memory: {mem['system_memory_mb']:.0f}MB used, {mem['system_memory_available_mb']:.0f}MB available")
            if 'gpu_memory_mb' in mem:
                print(f"GPU memory: {mem['gpu_memory_mb']:.0f}MB allocated, {mem['gpu_memory_cached_mb']:.0f}MB cached")
        
        print("=====================================\n") 