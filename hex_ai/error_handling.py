import os
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def check_data_loading_errors(files_attempted: int, files_with_errors: int, error_details: List[Tuple[str, str]], error_log_dir: str):
    """
    Checks error statistics after data loading. If thresholds are exceeded, writes an error log and raises RuntimeError.
    Args:
        files_attempted: Number of files attempted
        files_with_errors: Number of files with errors
        error_details: List of (filename, error_message)
        error_log_dir: Directory to write error.log
    """
    if files_attempted > 5 and files_with_errors > 2 and files_with_errors / files_attempted > 0.10:
        os.makedirs(error_log_dir, exist_ok=True)
        log_path = os.path.join(error_log_dir, "error.log")
        with open(log_path, "w") as f:
            f.write(f"Data loading error summary:\n")
            f.write(f"Too many data loading errors\n")
            f.write(f"Files attempted: {files_attempted}\n")
            f.write(f"Files with errors: {files_with_errors}\n")
            f.write(f"Error rate: {files_with_errors / files_attempted:.2%}\n\n")
            f.write("Details:\n")
            for filename, errmsg in error_details:
                f.write(f"{filename}: {errmsg}\n")
        raise RuntimeError(
            f"Too many data loading errors: {files_with_errors} out of {files_attempted} files failed (>10%). "
            f"See error log at {log_path} for details."
        )


class BoardStateErrorTracker:
    """Tracks board state validation errors during training."""
    
    def __init__(self, max_errors: int = 5, max_error_rate: float = 0.05):
        """
        Initialize error tracker.
        
        Args:
            max_errors: Maximum number of errors before failing
            max_error_rate: Maximum error rate (errors/total_samples) before failing
        """
        self.max_errors = max_errors
        self.max_error_rate = max_error_rate
        self.error_count = 0
        self.total_samples = 0
        self.error_details = []
        
    def record_error(self, board_state: Optional[object] = None, error_msg: str = "", 
                    file_info: str = "", sample_info: str = ""):
        """
        Record a board state validation error.
        
        Args:
            board_state: The problematic board state (optional, for debugging)
            error_msg: The error message
            file_info: Information about the file being processed
            sample_info: Information about the specific sample
        """
        self.error_count += 1
        self.total_samples += 1
        
        # Store error details for logging
        error_detail = {
            'error_count': self.error_count,
            'error_msg': error_msg,
            'file_info': file_info,
            'sample_info': sample_info,
            'board_state_info': str(board_state) if board_state is not None else "None"
        }
        self.error_details.append(error_detail)
        
        # Compact error logging
        import logging
        if logging.getLogger().getEffectiveLevel() <= logging.WARNING:
            print("E", end="", flush=True)
        
        # Detailed logging to file only
        logger.debug(f"Board state validation error #{self.error_count}: {error_msg}")
        if file_info:
            logger.debug(f"File: {file_info}")
        if sample_info:
            logger.debug(f"Sample: {sample_info}")
        
        # Check if we should fail
        self._check_error_thresholds()
    
    def record_success(self):
        """Record a successful sample processing."""
        self.total_samples += 1
    
    def _check_error_thresholds(self):
        """Check if error thresholds are exceeded and fail if necessary."""
        if self.total_samples < 10:
            return  # Need at least 10 samples for meaningful error rate
        
        error_rate = self.error_count / self.total_samples
        
        if (self.error_count > self.max_errors and 
            error_rate > self.max_error_rate):
            
            # Write error log
            self._write_error_log()
            
            # Raise exception
            raise RuntimeError(
                f"Too many board state validation errors: {self.error_count} errors out of "
                f"{self.total_samples} samples ({error_rate:.2%}). "
                f"Thresholds: max_errors={self.max_errors}, max_error_rate={self.max_error_rate:.1%}. "
                f"See error log for details."
            )
    
    def _write_error_log(self):
        """Write detailed error log."""
        try:
            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            log_path = log_dir / "board_state_errors.log"
            with open(log_path, "w") as f:
                f.write("Board State Validation Error Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total samples processed: {self.total_samples}\n")
                f.write(f"Total errors: {self.error_count}\n")
                f.write(f"Error rate: {self.error_count / self.total_samples:.2%}\n")
                f.write(f"Max errors threshold: {self.max_errors}\n")
                f.write(f"Max error rate threshold: {self.max_error_rate:.1%}\n\n")
                
                f.write("Error Details:\n")
                f.write("-" * 30 + "\n")
                for i, detail in enumerate(self.error_details, 1):
                    f.write(f"Error #{i}:\n")
                    f.write(f"  Message: {detail['error_msg']}\n")
                    f.write(f"  File: {detail['file_info']}\n")
                    f.write(f"  Sample: {detail['sample_info']}\n")
                    f.write(f"  Board state: {detail['board_state_info']}\n")
                    f.write("\n")
            
            logger.error(f"Board state validation errors logged to {log_path}")
            
        except Exception as e:
            logger.error(f"Failed to write error log: {e}")
    
    def get_stats(self) -> Dict[str, float]:
        """Get current error statistics."""
        if self.total_samples == 0:
            return {'error_count': 0, 'total_samples': 0, 'error_rate': 0.0}
        
        return {
            'error_count': self.error_count,
            'total_samples': self.total_samples,
            'error_rate': self.error_count / self.total_samples
        }


# Global error tracker instance
_board_state_error_tracker = BoardStateErrorTracker()


def get_board_state_error_tracker() -> BoardStateErrorTracker:
    """Get the global board state error tracker."""
    return _board_state_error_tracker


def reset_board_state_error_tracker():
    """Reset the global board state error tracker."""
    global _board_state_error_tracker
    _board_state_error_tracker = BoardStateErrorTracker() 