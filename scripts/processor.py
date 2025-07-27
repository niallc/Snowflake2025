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
    Sequential processor for debugging and testing.
    
    This provides the same interface as ParallelProcessor but processes
    files one at a time, making debugging easier.
    """
    
    def process_files(self, trmph_files: List[Path], config: ProcessingConfig) -> List[Dict]:
        """Process files sequentially."""
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
            
            result = process_single_file_worker(file_info)
            results.append(result)
            
            if result['success']:
                logger.info(f"✓ Processed {file_path}")
            else:
                logger.error(f"✗ Failed to process {file_path}: {result['error']}")
        
        return results


class TRMPHProcessor:
    """
    Main processing orchestrator that handles both sequential and parallel processing.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
        # Validate configuration
        self.config.validate()
        
        # Ensure output directory exists
        if not self.config.output_dir.exists():
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Choose processor based on configuration
        if self.config.max_workers > 1:
            self.processor = ParallelProcessor(self.config.max_workers)
        else:
            self.processor = SequentialProcessor()
    
    def process_all_files(self) -> List[Dict]:
        """
        Process all TRMPH files according to configuration.
        
        Returns:
            List of processing results
        """
        # Find files to process
        trmph_files = self._find_trmph_files()
        
        if not trmph_files:
            logger.warning("No TRMPH files found to process")
            return []
        
        logger.info(f"Found {len(trmph_files)} files to process")
        
        # Process files
        results = self.processor.process_files(trmph_files, self.config)
        
        # Generate summary
        self._log_summary(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _find_trmph_files(self) -> List[Path]:
        """Find all TRMPH files to process."""
        trmph_files = list(self.config.data_dir.rglob("*.trmph"))
        
        if self.config.max_files:
            trmph_files = trmph_files[:self.config.max_files]
        
        return trmph_files
    
    def _log_summary(self, results: List[Dict]):
        """Log processing summary."""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        logger.info("")
        logger.info("Processing Summary:")
        logger.info(f"  Total files: {len(results)}")
        logger.info(f"  Successful: {len(successful)}")
        logger.info(f"  Failed: {len(failed)}")
        logger.info(f"  Success rate: {len(successful)/len(results)*100:.1f}%")
        
        if failed:
            logger.info("")
            logger.info("Failed files:")
            for result in failed:
                logger.info(f"  - {result['file_path']}: {result['error']}")
    
    def _save_results(self, results: List[Dict]):
        """Save processing results to file."""
        stats_file = self.config.output_dir / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Processing statistics saved to: {stats_file}") 