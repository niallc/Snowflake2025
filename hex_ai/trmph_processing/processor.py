"""
Processing orchestrators for TRMPH files.

This module provides both parallel and sequential processing capabilities
with a clean interface and proper error handling.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
from pathlib import Path
import json

from .config import ProcessingConfig
from .workers import process_single_file_worker

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """
    Handles parallel processing of TRMPH files.
    
    This class encapsulates all multiprocessing logic and provides
    a clean interface for processing files in parallel.
    """
    
    def __init__(self, max_workers: int = 6):
        self.max_workers = max_workers
        self.results = []
    
    def process_files(self, trmph_files: List[Path], config: ProcessingConfig) -> List[Dict]:
        """
        Process files in parallel using ProcessPoolExecutor.
        
        Args:
            trmph_files: List of TRMPH file paths to process
            config: Processing configuration
            
        Returns:
            List of processing results (one per file)
        """
        # Prepare file info for workers
        file_infos = self._prepare_file_infos(trmph_files, config)
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = self._submit_jobs(executor, file_infos)
            results = self._collect_results(futures)
        
        return results
    
    def _prepare_file_infos(self, trmph_files: List[Path], config: ProcessingConfig) -> List[Dict]:
        """Prepare file information for worker processes."""
        file_infos = []
        
        for i, file_path in enumerate(trmph_files):
            file_info = {
                'file_path': str(file_path),
                'file_idx': i,
                'data_dir': str(config.data_dir),
                'output_dir': str(config.output_dir),
                'run_tag': config.run_tag,
                'position_selector': config.position_selector
            }
            file_infos.append(file_info)
        
        return file_infos
    
    def _submit_jobs(self, executor: ProcessPoolExecutor, file_infos: List[Dict]):
        """Submit jobs to the executor."""
        futures = {}
        
        for file_info in file_infos:
            future = executor.submit(process_single_file_worker, file_info)
            futures[future] = file_info
        
        return futures
    
    def _collect_results(self, futures: Dict) -> List[Dict]:
        """Collect results from completed futures."""
        results = []
        
        for future in as_completed(futures):
            file_info = futures[future]
            
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    logger.info(f"✓ Processed {file_info['file_path']}")
                else:
                    logger.error(f"✗ Failed to process {file_info['file_path']}: {result['error']}")
                    
            except Exception as e:
                # Handle unexpected errors in the future itself
                error_result = {
                    'success': False,
                    'error': f"Unexpected error: {str(e)}",
                    'file_path': file_info['file_path'],
                    'file_idx': file_info['file_idx']
                }
                results.append(error_result)
                logger.error(f"✗ Unexpected error processing {file_info['file_path']}: {e}")
        
        return results


class SequentialProcessor:
    """
    Handles sequential processing of TRMPH files.
    
    This class provides the same interface as ParallelProcessor but
    processes files one at a time, useful for debugging.
    """
    
    def process_files(self, trmph_files: List[Path], config: ProcessingConfig) -> List[Dict]:
        """
        Process files sequentially.
        
        Args:
            trmph_files: List of TRMPH file paths to process
            config: Processing configuration
            
        Returns:
            List of processing results (one per file)
        """
        results = []
        
        for i, file_path in enumerate(trmph_files):
            file_info = {
                'file_path': str(file_path),
                'file_idx': i,
                'data_dir': str(config.data_dir),
                'output_dir': str(config.output_dir),
                'run_tag': config.run_tag,
                'position_selector': config.position_selector
            }
            
            try:
                result = process_single_file_worker(file_info)
                results.append(result)
                
                if result['success']:
                    logger.info(f"✓ Processed {file_path}")
                else:
                    logger.error(f"✗ Failed to process {file_path}: {result['error']}")
                    
            except Exception as e:
                error_result = {
                    'success': False,
                    'error': f"Unexpected error: {str(e)}",
                    'file_path': str(file_path),
                    'file_idx': i
                }
                results.append(error_result)
                logger.error(f"✗ Unexpected error processing {file_path}: {e}")
        
        return results


class TRMPHProcessor:
    """
    Main orchestrator for TRMPH file processing.
    
    This class coordinates the discovery of TRMPH files and delegates
    processing to either parallel or sequential processors.
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize the processor with configuration.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.config.validate()
        
        # Create output directory if it doesn't exist
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Choose processor based on worker count
        if self.config.max_workers == 1:
            self.processor = SequentialProcessor()
        else:
            self.processor = ParallelProcessor(self.config.max_workers)
    
    def process_all_files(self) -> List[Dict]:
        """
        Process all TRMPH files in the data directory.
        
        Returns:
            List of processing results
        """
        # Find all TRMPH files
        trmph_files = self._find_trmph_files()
        
        if not trmph_files:
            logger.warning(f"No .trmph files found in {self.config.data_dir}")
            return []
        
        # Limit files if specified
        if self.config.max_files:
            trmph_files = trmph_files[:self.config.max_files]
            logger.info(f"Limited to {len(trmph_files)} files for testing")
        
        logger.info(f"Found {len(trmph_files)} .trmph files to process")
        
        # Process files
        results = self.processor.process_files(trmph_files, self.config)
        
        # Log summary and save results
        self._log_summary(results)
        self._save_results(results)
        
        return results
    
    def _find_trmph_files(self) -> List[Path]:
        """Find all .trmph files in the data directory."""
        trmph_files = list(self.config.data_dir.glob("**/*.trmph"))
        trmph_files.sort()  # Ensure consistent ordering
        return trmph_files
    
    def _log_summary(self, results: List[Dict]):
        """Log a summary of processing results."""
        total_files = len(results)
        successful_files = sum(1 for r in results if r['success'])
        failed_files = total_files - successful_files
        
        total_examples = sum(
            r.get('stats', {}).get('examples_generated', 0) 
            for r in results if r['success']
        )
        
        logger.info("")
        logger.info("PROCESSING SUMMARY:")
        logger.info(f"  Total files: {total_files}")
        logger.info(f"  Successful: {successful_files}")
        logger.info(f"  Failed: {failed_files}")
        logger.info(f"  Total examples generated: {total_examples}")
        
        if failed_files > 0:
            logger.warning(f"  {failed_files} files failed to process")
    
    def _save_results(self, results: List[Dict]):
        """Save processing results to JSON file."""
        results_file = self.config.output_dir / "processing_stats.json"
        
        # Convert Path objects to strings for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            if 'stats' in serializable_result:
                # Ensure stats are JSON serializable
                stats = serializable_result['stats']
                if 'file_path' in stats:
                    stats['file_path'] = str(stats['file_path'])
            serializable_results.append(serializable_result)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Processing statistics saved to {results_file}") 