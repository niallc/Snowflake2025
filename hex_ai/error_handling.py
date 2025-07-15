import os
from typing import List, Tuple

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