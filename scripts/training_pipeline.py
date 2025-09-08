#!/usr/bin/env python3
"""
Training Pipeline for Hex AI

A comprehensive pipeline that orchestrates:
1. Multi-worker self-play data generation
2. Data preprocessing and cleaning
3. TRMPH processing into training positions
4. Data shuffling and preparation
5. Model training with hyperparameter tuning

This replaces the previous bash-based approach with a proper Python pipeline
that provides better error handling, progress tracking, and configurability.
"""

import argparse
import logging
import os
import sys
import time
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

# Environment validation is now handled automatically in hex_ai/__init__.py
import hex_ai
from hex_ai.selfplay.selfplay_engine import SelfPlayEngine
from hex_ai.trmph_processing.cli import create_config_from_args, process_files
from hex_ai.file_utils import GracefulShutdown
from hex_ai.error_handling import GracefulShutdownRequested
from hex_ai.training_orchestration import run_hyperparameter_tuning_current_data

# Script imports (moved to top level)
from hex_ai.data_collection import combine_and_clean_files, collect_and_organize_data, parse_shard_ranges
from scripts.shuffle_processed_data import DataShuffler


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""
    
    # Model configuration
    model_path: str
    model_epoch: int = 4
    model_mini: int = 1
    
    # Self-play configuration
    num_games: int = 100000
    num_workers: int = 3  # Number of self-play workers
    search_widths: List[int] = field(default_factory=lambda: [13, 8])
    temperature: float = 1.5
    batch_size: int = 128
    cache_size: int = 60000
    
    # Data directories
    base_data_dir: str = "data"
    data_sources: List[str] = field(default_factory=lambda: [str(d) for d in hex_ai.data_config.DEFAULT_PROCESSED_DATA_DIRS])
    shard_ranges: List[str] = field(default_factory=lambda: ["all"])
    selfplay_dir: Optional[str] = None  # If provided, use existing raw self-play data
    
    # Processing configuration
    chunk_size: int = 10000
    position_selector: str = "all"
    max_workers_trmph: int = 6
    num_buckets_shuffle: int = 100
    
    # Training configuration
    max_samples: int = 35000000
    max_validation_samples: int = 137000
    results_dir: str = "checkpoints/hyperparameter_tuning"
    
    # Pipeline control
    run_game_collection: bool = False  # New: Collect games from multiple sources
    run_selfplay: bool = True
    run_preprocessing: bool = True
    run_trmph_processing: bool = True
    run_shuffling: bool = True
    run_training: bool = True
    cleanup_intermediate: bool = True
    
    def __post_init__(self):
        """Generate derived paths and validate configuration."""
        # Generate timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate model filename
        self.model_filename = f"epoch{self.model_epoch}_mini{self.model_mini}.pt.gz"
        self.model_full_path = os.path.join(self.model_path, self.model_filename)
        
        # Generate data directories for this run
        if self.selfplay_dir is None:
            self.selfplay_dir = str(Path(self.base_data_dir) / "sf25" / f"selfplay_{self.run_timestamp}")
        
        # Generate predictable output directory names based on input
        input_name = Path(self.selfplay_dir).name if self.selfplay_dir else f"run_{self.run_timestamp}"
        self.cleaned_dir = str(Path(self.base_data_dir) / "cleaned" / f"cleaned_{input_name}")
        self.processed_dir = str(Path(self.base_data_dir) / "processed" / f"processed_{input_name}")
        self.shuffled_dir = str(Path(self.base_data_dir) / "processed" / f"shuffled_{input_name}")
        self.temp_dir = str(Path(self.base_data_dir) / "processed" / f"temp_buckets_{input_name}")
    
    def validate(self, check_model: bool = True, check_data: bool = True):
        """Validate configuration (called when actually running the pipeline)."""
        # Validate model exists
        if check_model and not os.path.exists(self.model_full_path):
            raise FileNotFoundError(f"Model not found: {self.model_full_path}")
        
        # Validate existing data directories
        if check_data:
            for data_dir in self.data_sources:
                if not os.path.exists(data_dir):
                    raise FileNotFoundError(f"Data directory not found: {data_dir}")


class GameCollectionStep:
    """Handles collection of games from multiple sources."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run(self) -> str:
        """Run game collection and return the output directory."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 0: GAME COLLECTION FROM MULTIPLE SOURCES")
        self.logger.info("=" * 60)
        
        # Use default source directories from config
        source_dirs = [Path(d) for d in self.config.data_sources]
        output_dir = Path(self.config.selfplay_dir)
        
        self.logger.info(f"Source directories: {[str(d) for d in source_dirs]}")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Run game collection
        stats = collect_and_organize_data(source_dirs, output_dir, self.config.chunk_size)
        
        if "error" in stats:
            raise RuntimeError(f"Game collection failed: {stats['error']}")
        
        self.logger.info(f"Game collection completed successfully!")
        self.logger.info(f"Collected {stats['unique_games']} unique games from {stats['total_files']} files")
        
        return str(output_dir)


class SelfPlayStep:
    """Handles multi-worker self-play data generation."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run(self) -> str:
        """Run multi-worker self-play and return the output directory."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: MULTI-WORKER SELF-PLAY GENERATION")
        self.logger.info("=" * 60)
        
        # Create output directory
        Path(self.config.selfplay_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate games per worker
        games_per_worker = self.config.num_games // self.config.num_workers
        remaining_games = self.config.num_games % self.config.num_workers
        
        self.logger.info(f"Model: {self.config.model_full_path}")
        self.logger.info(f"Total games: {self.config.num_games}")
        self.logger.info(f"Workers: {self.config.num_workers}")
        self.logger.info(f"Games per worker: {games_per_worker}")
        self.logger.info(f"Output directory: {self.config.selfplay_dir}")
        
        # Create worker processes
        processes = []
        for worker_id in range(self.config.num_workers):
            # Calculate games for this worker
            worker_games = games_per_worker + (1 if worker_id < remaining_games else 0)
            
            # Create worker-specific directory
            worker_dir = str(Path(self.config.selfplay_dir) / f"worker_{worker_id}")
            Path(worker_dir).mkdir(parents=True, exist_ok=True)
            
            # Create process
            process = mp.Process(
                target=self._run_worker,
                args=(worker_id, worker_games, worker_dir)
            )
            processes.append(process)
        
        # Start all workers with 2-second delays to avoid timestamp collisions
        start_time = time.time()
        for i, process in enumerate(processes):
            process.start()
            if i < len(processes) - 1:  # Don't sleep after the last worker
                time.sleep(2)  # 2-second delay between workers
        
        # Wait for all workers to complete
        for i, process in enumerate(processes):
            process.join()
            if process.exitcode != 0:
                self.logger.error(f"Worker {i} failed with exit code {process.exitcode}")
                # Graceful shutdown of remaining processes
                self._handle_shutdown(processes[i+1:])
                raise RuntimeError(f"Self-play worker {i} failed")
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Self-play completed in {elapsed_time:.1f} seconds")
        
        return self.config.selfplay_dir
    
    def _run_worker(self, worker_id: int, num_games: int, output_dir: str):
        """Run a single self-play worker."""
        try:
            # Set worker-specific seed
            seed = 42 + worker_id * 1000
            
            # Create self-play engine
            engine = SelfPlayEngine(
                model_path=self.config.model_full_path,
                num_workers=1,  # Each process is a single worker
                batch_size=self.config.batch_size,
                cache_size=self.config.cache_size,
                search_widths=self.config.search_widths,
                temperature=self.config.temperature,
                verbose=1,
                streaming_save=True,
                use_batched_inference=True,
                output_dir=output_dir
            )
            
            # Generate games
            engine.generate_games_streaming(
                num_games=num_games,
                progress_interval=10
            )
            
        except Exception as e:
            self.logger.error(f"Worker {worker_id} failed: {e}")
            raise
        finally:
            # Clean up GPU memory
            self._cleanup_gpu_memory()
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory after self-play."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("GPU memory cleaned up")
        except ImportError:
            pass  # PyTorch not available
    
    def _handle_shutdown(self, processes):
        """Handle graceful shutdown of worker processes."""
        self.logger.info("Shutdown requested, terminating workers...")
        
        # Send SIGTERM to all processes
        for process in processes:
            if process.is_alive():
                process.terminate()
        
        # Wait briefly for graceful termination
        time.sleep(5)
        
        # Force kill if still alive
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=10)


class PreprocessingStep:
    """Handles self-play data preprocessing."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run(self, input_dir: str) -> str:
        """Preprocess self-play data and return cleaned directory."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: SELF-PLAY DATA PREPROCESSING")
        self.logger.info("=" * 60)
        

        
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {self.config.cleaned_dir}")
        self.logger.info(f"Chunk size: {self.config.chunk_size}")
        
        # Check if output already exists
        if Path(self.config.cleaned_dir).exists() and list(Path(self.config.cleaned_dir).glob("*.trmph")):
            raise FileExistsError(
                f"Output directory already exists and contains data: {self.config.cleaned_dir}\n"
                f"This suggests the data has already been processed. To avoid wasting compute time,\n"
                f"either:\n"
                f"1. Use a different --selfplay-dir\n"
                f"2. Remove the existing output directory\n"
                f"3. Use --no-preprocessing to skip this step"
            )
        
        # Create output directory
        Path(self.config.cleaned_dir).mkdir(parents=True, exist_ok=True)
        
        # Run preprocessing
        combine_and_clean_files(
            input_dir=Path(input_dir),
            output_dir=Path(self.config.cleaned_dir),
            chunk_size=self.config.chunk_size
        )
        
        return self.config.cleaned_dir


class TRMPHProcessingStep:
    """Handles TRMPH file processing into training positions."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run(self, input_dir: str) -> str:
        """Process TRMPH files and return processed directory."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: TRMPH PROCESSING")
        self.logger.info("=" * 60)
        
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {self.config.processed_dir}")
        self.logger.info(f"Position selector: {self.config.position_selector}")
        self.logger.info(f"Max workers: {self.config.max_workers_trmph}")
        
        # Check if output already exists
        if Path(self.config.processed_dir).exists() and list(Path(self.config.processed_dir).glob("*.pkl.gz")):
            raise FileExistsError(
                f"Output directory already exists and contains data: {self.config.processed_dir}\n"
                f"This suggests the data has already been processed. To avoid wasting compute time,\n"
                f"either:\n"
                f"1. Use a different --selfplay-dir\n"
                f"2. Remove the existing output directory\n"
                f"3. Use --no-trmph-processing to skip this step"
            )
        
        # Create output directory
        Path(self.config.processed_dir).mkdir(parents=True, exist_ok=True)
        
        # Create configuration
        config = create_config_from_args(type('Args', (), {
            'data_dir': input_dir,
            'output_dir': self.config.processed_dir,
            'max_files': None,
            'position_selector': self.config.position_selector,
            'run_tag': f"pipeline_{self.config.run_timestamp}",
            'max_workers': self.config.max_workers_trmph,
            'sequential': False
        })())
        
        # Process files
        results = process_files(config)
        
        self.logger.info(f"TRMPH processing completed: {results}")
        
        return self.config.processed_dir


class ShufflingStep:
    """Handles data shuffling."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run(self, input_dir: str) -> str:
        """Shuffle processed data and return shuffled directory."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: DATA SHUFFLING")
        self.logger.info("=" * 60)
        

        
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {self.config.shuffled_dir}")
        self.logger.info(f"Number of buckets: {self.config.num_buckets_shuffle}")
        
        # Check if output already exists
        if Path(self.config.shuffled_dir).exists() and list(Path(self.config.shuffled_dir).glob("*.pkl.gz")):
            raise FileExistsError(
                f"Output directory already exists and contains data: {self.config.shuffled_dir}\n"
                f"This suggests the data has already been processed. To avoid wasting compute time,\n"
                f"either:\n"
                f"1. Use a different --selfplay-dir\n"
                f"2. Remove the existing output directory\n"
                f"3. Use --no-shuffling to skip this step"
            )
        
        # Create output directory
        Path(self.config.shuffled_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Create shuffler
        shuffler = DataShuffler(
            input_dir=input_dir,
            output_dir=self.config.shuffled_dir,
            temp_dir=self.config.temp_dir,
            num_buckets=self.config.num_buckets_shuffle,
            resume_enabled=True,
            cleanup_temp=True,
            validation_enabled=True
        )
        
        # Run shuffling
        shuffler.shuffle_data()
        
        return self.config.shuffled_dir


class TrainingStep:
    """Handles model training."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run(self, new_shuffled_dir: str):
        """Run training with the new data."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: MODEL TRAINING")
        self.logger.info("=" * 60)
        
        # Create results directory
        results_dir = str(Path(self.config.results_dir) / f"pipeline_{self.config.run_timestamp}")
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"New data directory: {new_shuffled_dir}")
        self.logger.info(f"Existing data directories: {self.config.data_sources}")
        self.logger.info(f"Shard ranges: {self.config.shard_ranges}")
        self.logger.info(f"Results directory: {results_dir}")
        self.logger.info(f"Max samples: {self.config.max_samples}")
        self.logger.info(f"Resume from: {self.config.model_full_path}")
        
        # Create shutdown handler
        shutdown_handler = GracefulShutdown()
        
        # Create experiment configurations from sweep
        from scripts.hyperparam_sweep import SWEEP, all_param_combinations, make_experiment_name
        
        all_configs = list(all_param_combinations(SWEEP))
        experiments = []
        for i, config in enumerate(all_configs):
            # Compute value_weight so that policy_weight + value_weight = 1
            config = dict(config)  # Make a copy to avoid mutating the sweep dict
            if "policy_weight" in config:
                config["value_weight"] = 1.0 - config["policy_weight"]
            exp_name = make_experiment_name(config, i, tag="pipeline_sweep")
            experiments.append({
                'experiment_name': exp_name,
                'hyperparameters': config
            })
        
        # Run training
        if new_shuffled_dir:
            all_data_dirs = [new_shuffled_dir] + self.config.data_sources
            all_shard_ranges = ["all"] + self.config.shard_ranges  # "all" for new data
        else:
            all_data_dirs = self.config.data_sources
            all_shard_ranges = self.config.shard_ranges
        
        results = run_hyperparameter_tuning_current_data(
            experiments=experiments,
            data_dirs=all_data_dirs,
            results_dir=results_dir,
            train_ratio=0.8,
            num_epochs=2,  # Default from hyperparam_sweep
            early_stopping_patience=None,
            random_seed=42,
            max_examples_unaugmented=self.config.max_samples,
            max_validation_examples=self.config.max_validation_samples,
            experiment_name=None,
            enable_augmentation=True,
            mini_epoch_samples=250000,  # Default from hyperparam_sweep
            resume_from=self.config.model_full_path,
            shard_ranges=all_shard_ranges,
            shutdown_handler=shutdown_handler,
            run_timestamp=self.config.run_timestamp,
            override_checkpoint_hyperparameters=False,
            shuffle_shards=True
        )
        
        self.logger.info(f"Training completed: {results}")
        return results_dir


class TrainingPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.shutdown_handler = GracefulShutdown()
        
        # Initialize steps
        self.game_collection_step = GameCollectionStep(config)
        self.selfplay_step = SelfPlayStep(config)
        self.preprocessing_step = PreprocessingStep(config)
        self.trmph_step = TRMPHProcessingStep(config)
        self.shuffling_step = ShufflingStep(config)
        self.training_step = TrainingStep(config)
        
        # Track progress
        self.current_step = 0
        self.step_results = {}
    
    def run(self):
        """Run the complete pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("HEX AI TRAINING PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Run timestamp: {self.config.run_timestamp}")
        self.logger.info(f"Model: {self.config.model_full_path}")
        self.logger.info(f"Configuration: {self.config}")
        
        # Validate configuration
        self.config.validate(
            check_model=self.config.run_selfplay or self.config.run_training,
            check_data=self.config.run_training
        )
        
        start_time = time.time()
        
        try:
            # Step 0: Game collection (optional)
            if self.config.run_game_collection:
                self.current_step = 0
                self.logger.info(f"\nStarting step {self.current_step}: Game collection")
                collected_dir = self.game_collection_step.run()
                self.step_results['game_collection'] = collected_dir
                # Use collected data as selfplay input
                selfplay_dir = collected_dir
            else:
                selfplay_dir = None
            
            # Step 1: Self-play (optional)
            if self.config.run_selfplay:
                self.current_step = 1
                self.logger.info(f"\nStarting step {self.current_step}: Self-play")
                selfplay_dir = self.selfplay_step.run()
                self.step_results['selfplay'] = selfplay_dir
            elif self.config.selfplay_dir is not None:
                self.logger.info(f"Using existing self-play directory: {self.config.selfplay_dir}")
                selfplay_dir = self.config.selfplay_dir
                self.step_results['selfplay'] = selfplay_dir
            else:
                self.logger.info("Skipping self-play (disabled)")
                selfplay_dir = None
            
            # Step 2: Preprocessing (optional)
            if self.config.run_preprocessing and selfplay_dir:
                self.current_step = 2
                self.logger.info(f"\nStarting step {self.current_step}: Preprocessing")
                cleaned_dir = self.preprocessing_step.run(selfplay_dir)
                self.step_results['preprocessing'] = cleaned_dir
            else:
                self.logger.info("Skipping preprocessing (disabled or no self-play data)")
                cleaned_dir = None
            
            # Step 3: TRMPH processing (optional)
            if self.config.run_trmph_processing and cleaned_dir:
                self.current_step = 3
                self.logger.info(f"\nStarting step {self.current_step}: TRMPH processing")
                processed_dir = self.trmph_step.run(cleaned_dir)
                self.step_results['trmph_processing'] = processed_dir
            else:
                self.logger.info("Skipping TRMPH processing (disabled or no cleaned data)")
                processed_dir = None
            
            # Step 4: Shuffling (optional)
            if self.config.run_shuffling and processed_dir:
                self.current_step = 4
                self.logger.info(f"\nStarting step {self.current_step}: Shuffling")
                shuffled_dir = self.shuffling_step.run(processed_dir)
                self.step_results['shuffling'] = shuffled_dir
            else:
                self.logger.info("Skipping shuffling (disabled or no processed data)")
                shuffled_dir = None
            
            # Step 5: Training (optional)
            if self.config.run_training:
                self.current_step = 5
                self.logger.info(f"\nStarting step {self.current_step}: Training")
                
                # Use newly shuffled data if available, otherwise use existing data sources
                if shuffled_dir:
                    training_data_dir = shuffled_dir
                    self.logger.info(f"Using newly shuffled data: {training_data_dir}")
                elif self.config.data_sources:
                    # When no new shuffled data, just use existing data sources (no duplication)
                    training_data_dir = None
                    self.logger.info(f"Using existing data sources: {self.config.data_sources}")
                else:
                    self.logger.error("No training data available")
                    raise ValueError("No training data available - need either shuffled data or data sources")
                
                results_dir = self.training_step.run(training_data_dir)
                self.step_results['training'] = results_dir
            else:
                self.logger.info("Skipping training (disabled)")
                results_dir = None
            
            # Cleanup intermediate files
            if self.config.cleanup_intermediate:
                self._cleanup_intermediate_files()
            
            # Final summary
            elapsed_time = time.time() - start_time
            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
            self.logger.info(f"Results: {self.step_results}")
            
        except GracefulShutdownRequested:
            self.logger.info("Pipeline interrupted by graceful shutdown request")
            raise
        except Exception as e:
            self.logger.error(f"Pipeline failed at step {self.current_step}: {e}")
            self.logger.error("Step results so far:")
            for step, result in self.step_results.items():
                self.logger.error(f"  {step}: {result}")
            raise
    
    def _cleanup_intermediate_files(self):
        """Clean up intermediate files to save disk space."""
        self.logger.info("Cleaning up intermediate files...")
        
        # Remove temporary directories
        temp_dirs = [
            self.config.selfplay_dir,
            self.config.cleaned_dir,
            self.config.processed_dir,
            self.config.temp_dir
        ]
        
        for temp_dir in temp_dirs:
            temp_path = Path(temp_dir)
            if temp_path.exists():
                try:
                    import shutil
                    shutil.rmtree(temp_path)
                    self.logger.info(f"Removed: {temp_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {temp_dir}: {e}")


def setup_logging():
    """Setup logging for the pipeline."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'pipeline_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hex AI Training Pipeline",
        epilog="""
Examples:
  # Run complete pipeline with default settings
  python scripts/training_pipeline.py --model-path checkpoints/experiment/epoch4_mini1.pt.gz
  
  # Use current best model from model_config.py
  python scripts/training_pipeline.py --use-current-best-model
  
  # Run with game collection from multiple sources
  python scripts/training_pipeline.py --use-current-best-model --run-game-collection --no-selfplay
  
  # Run only self-play and preprocessing
  python scripts/training_pipeline.py --use-current-best-model --no-training --no-shuffling --no-trmph-processing
  
  # Run with custom settings
  python scripts/training_pipeline.py --use-current-best-model --num-games 50000 --num-workers 5 --temperature 1.0
  
  # Use multiple data directories with specific shard ranges
  python scripts/training_pipeline.py --use-current-best-model --data_dirs data/processed/sf18_shuffled data/processed/shuffled_sf25_20250906 --shard_ranges "251-300" "all"
  
  # Use existing raw self-play data
  python scripts/training_pipeline.py --use-current-best-model --selfplay-dir data/sf25/aug_04 --no-selfplay
  
  # Complete pipeline with game collection and training
  python scripts/training_pipeline.py --use-current-best-model --run-game-collection --no-selfplay --no-preprocessing
        """
    )
    
    # Model configuration
    parser.add_argument("--model-path", help="Path to model checkpoint directory")
    parser.add_argument("--model-epoch", type=int, default=4, help="Model epoch number")
    parser.add_argument("--model-mini", type=int, default=1, help="Model mini-epoch number")
    parser.add_argument("--use-current-best-model", action="store_true", 
                       help="Use current best model from hex_ai.inference.model_config")
    
    # Self-play configuration
    parser.add_argument("--num-games", type=int, default=100000, help="Number of games to generate")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of self-play workers")
    parser.add_argument("--search-widths", type=int, nargs='+', default=[13, 8], help="Search widths for minimax")
    parser.add_argument("--temperature", type=float, default=1.5, help="Temperature for move sampling")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--cache-size", type=int, default=60000, help="Cache size for model inference")
    
    # Data configuration
    parser.add_argument("--base-data-dir", default="data", help="Base directory for data")
    parser.add_argument("--selfplay-dir", help="Use existing raw self-play directory (skip self-play generation)")
    parser.add_argument("--data_dirs", type=str, nargs='+', default=["data/processed/shuffled"], 
                       help="Existing data directories to use for training")
    parser.add_argument("--shard_ranges", type=str, nargs='+',
                       help='Shard ranges for each data directory. Format: "start-end" or "all" (e.g., --shard_ranges "251-300" "all" to use shards 251-300 from first dir, all shards from second).')
    parser.add_argument("--chunk-size", type=int, default=10000, help="Chunk size for preprocessing")
    parser.add_argument("--position-selector", default="all", choices=["all", "final", "penultimate"], help="Position selector for TRMPH processing")
    parser.add_argument("--max-workers-trmph", type=int, default=6, help="Max workers for TRMPH processing")
    parser.add_argument("--num-buckets-shuffle", type=int, default=100, help="Number of buckets for shuffling")
    
    # Training configuration
    parser.add_argument("--max-samples", type=int, default=35000000, help="Max training samples")
    parser.add_argument("--max-validation-samples", type=int, default=137000, help="Max validation samples")
    parser.add_argument("--results-dir", default="checkpoints/hyperparameter_tuning", help="Results directory")
    
    # Pipeline control
    parser.add_argument("--run-game-collection", action="store_true", help="Run game collection from multiple sources")
    parser.add_argument("--no-selfplay", action="store_true", help="Skip self-play step")
    parser.add_argument("--no-preprocessing", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--no-trmph-processing", action="store_true", help="Skip TRMPH processing step")
    parser.add_argument("--no-shuffling", action="store_true", help="Skip shuffling step")
    parser.add_argument("--no-training", action="store_true", help="Skip training step")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep intermediate files")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Handle current best model option
        if args.use_current_best_model:
            if args.model_path:
                raise ValueError("Cannot use both --model-path and --use-current-best-model. Choose one.")
            
            try:
                from hex_ai.inference.model_config import get_model_path, get_model_dir
                model_path = get_model_path("current_best")
                args.model_path = get_model_dir("current_best")
                
                # Extract epoch and mini from the filename
                import os
                filename = os.path.basename(model_path)
                # Expected format: epoch2_mini201.pt.gz
                if 'epoch' in filename and 'mini' in filename:
                    parts = filename.split('_')
                    for part in parts:
                        if part.startswith('epoch'):
                            args.model_epoch = int(part[5:])
                        elif part.startswith('mini'):
                            args.model_mini = int(part[4:].split('.')[0])
                
                logger.info(f"Using current best model: {model_path}")
                logger.info(f"Model directory: {args.model_path}")
                logger.info(f"Model epoch: {args.model_epoch}, mini: {args.model_mini}")
            except ImportError:
                raise ValueError("Could not import hex_ai.inference.model_config")
            except Exception as e:
                raise ValueError(f"Could not get current best model path: {e}")
        elif not args.model_path:
            raise ValueError("Must specify either --model-path or --use-current-best-model")
        
        # Validate data directories and shard ranges
        if args.shard_ranges and len(args.shard_ranges) != len(args.data_dirs):
            raise ValueError(f"Number of shard ranges ({len(args.shard_ranges)}) must match number of data directories ({len(args.data_dirs)})")

        # Create configuration
        config = PipelineConfig(
            model_path=args.model_path,
            model_epoch=args.model_epoch,
            model_mini=args.model_mini,
            num_games=args.num_games,
            num_workers=args.num_workers,
            search_widths=args.search_widths,
            temperature=args.temperature,
            batch_size=args.batch_size,
            cache_size=args.cache_size,
            base_data_dir=args.base_data_dir,
            data_sources=args.data_dirs,
            shard_ranges=args.shard_ranges,
            selfplay_dir=args.selfplay_dir,
            chunk_size=args.chunk_size,
            position_selector=args.position_selector,
            max_workers_trmph=args.max_workers_trmph,
            num_buckets_shuffle=args.num_buckets_shuffle,
            max_samples=args.max_samples,
            max_validation_samples=args.max_validation_samples,
            results_dir=args.results_dir,
            run_game_collection=args.run_game_collection,
            run_selfplay=not args.no_selfplay and args.selfplay_dir is None,
            run_preprocessing=not args.no_preprocessing,
            run_trmph_processing=not args.no_trmph_processing,
            run_shuffling=not args.no_shuffling,
            run_training=not args.no_training,
            cleanup_intermediate=not args.no_cleanup
        )
        
        # Create and run pipeline
        pipeline = TrainingPipeline(config)
        pipeline.run()
        
        logger.info(f"Pipeline completed successfully. Log file: {log_file}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    # Removed generic Exception handler to allow proper stack traces for debugging


if __name__ == "__main__":
    main()