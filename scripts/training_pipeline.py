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
import json
import logging
import os
import re
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
from hex_ai.config import DEFAULT_CACHE_SIZE, DEFAULT_TEMPERATURE_START
from hex_ai.selfplay.selfplay_engine import SelfPlayEngine
from hex_ai.trmph_processing.cli import create_config_from_args, process_files
from hex_ai.file_utils import GracefulShutdown
from hex_ai.error_handling import GracefulShutdownRequested
from hex_ai.training_orchestration import run_hyperparameter_tuning_current_data
from hex_ai.training_utils import create_hyperparameter_sweep, HYPERPARAMETER_SHORT_LABELS

# Script imports (moved to top level)
from hex_ai.data_collection import combine_and_clean_files, collect_and_organize_data
from hex_ai.validation_defaults import resolve_validation_config, log_validation_summary
from hex_ai.data_pipeline import DataShuffler
from hex_ai.memory_profiler import start_profiling, take_snapshot, stop_profiling


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""
    
    # Model configuration
    model_path: str
    model_epoch: int  # No default - must be explicitly set
    model_mini: int  # No default - must be explicitly set
    
    # Self-play configuration
    num_games: int = 100000
    num_workers: int = 3  # Number of self-play workers
    temperature: float = DEFAULT_TEMPERATURE_START
    batch_size: int = 128
    cache_size: int = DEFAULT_CACHE_SIZE
    write_provenance: bool = True
    
    # Data directories - explicit types to avoid confusion
    base_data_dir: str = "data"
    raw_trmph_data_dirs: List[str] = field(default_factory=lambda: [str(d) for d in hex_ai.data_config.DEFAULT_SOURCE_DIRS])  # Raw .trmph files to collect
    cleaned_trmph_data_dirs: List[str] = field(default_factory=list)  # Already cleaned .trmph files
    ordered_positions_dirs: List[str] = field(default_factory=list)  # Existing ordered positions (not shuffled)
    training_data_dirs: List[str] = field(default_factory=lambda: [str(d) for d in hex_ai.data_config.DEFAULT_TRAINING_DATA_DIRS])  # Existing training data (shuffled positions)
    shard_ranges: List[str] = field(default_factory=lambda: ["all"])
    validation_dirs: Optional[List[str]] = None  # Validation data directories (optional, defaults to hardcoded values)
    validation_shard_ranges: Optional[List[str]] = None  # Validation shard ranges (optional, defaults to hardcoded values)
    no_validation: bool = False  # Disable validation entirely
    selfplay_dir: Optional[str] = None  # If provided, use existing raw self-play data
    
    # Processing configuration
    chunk_size: int = 10000
    position_selector: str = "all"
    policy_provenance_mode: str = "require"
    max_workers_trmph: int = 6
    num_buckets_shuffle: int = 100
    
    # Training configuration
    max_samples: int = 35000000
    max_validation_samples: int = 137000
    results_dir: str = "checkpoints/hyperparameter_tuning"
    override_checkpoint_hyperparameters: bool = False
    hyperparameter_overrides: Dict = field(default_factory=dict)
    restart_every_mini_epochs: int = 10
    max_mini_epochs_per_run: Optional[int] = None
    resume_mode: str = "next_epoch"
    target_end_epoch: Optional[int] = None
    allow_missing_stream_sidecar_fallback: bool = False
    internal_training_chunk_run: bool = False
    run_timestamp_override: Optional[str] = None
    
    # Pipeline control
    run_game_collection: bool = False  # New: Collect games from multiple sources
    run_selfplay: bool = True
    run_preprocessing: bool = True
    run_trmph_processing: bool = True
    run_shuffling: bool = True
    run_training: bool = True
    cleanup_intermediate: bool = True
    enable_memory_profiling: bool = False  # Enable memory profiling
    memory_profile_interval_seconds: int = 60  # Timeline sampling interval when profiling
    
    def __post_init__(self):
        """Generate derived paths and validate configuration."""
        # Generate timestamp for this run
        self.run_timestamp = (
            self.run_timestamp_override
            if self.run_timestamp_override
            else datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        
        # Generate model filename
        self.model_filename = f"epoch{self.model_epoch}_mini{self.model_mini}.pt.gz"
        self.model_full_path = os.path.join(self.model_path, self.model_filename)
        
        # Generate data directories for this run
        if self.selfplay_dir is None:
            self.selfplay_dir = str(Path(self.base_data_dir) / "sf25" / f"selfplay_{self.run_timestamp}")
        
        # Generate predictable output directory names based on input
        input_name = Path(self.selfplay_dir).name if self.selfplay_dir else f"run_{self.run_timestamp}"
        self.cleaned_dir = str(Path(self.base_data_dir) / "cleaned" / f"cleaned_{input_name}")
        self.ordered_positions_dir = str(Path(self.base_data_dir) / "processed" / f"ordered_positions_{input_name}")  # Individual positions (not shuffled)
        self.shuffled_dir = str(Path(self.base_data_dir) / "processed" / f"shuffled_{input_name}")  # Final training data (shuffled positions)
        self.temp_dir = str(Path(self.base_data_dir) / "processed" / f"temp_buckets_{input_name}")
    
    def validate(self, check_model: bool = True, check_data: bool = True):
        """Validate configuration (called when actually running the pipeline)."""
        # Validate model exists
        if check_model and not os.path.exists(self.model_full_path):
            raise FileNotFoundError(f"Model not found: {self.model_full_path}")
        
        # Validate data directories exist
        if check_data:
            for data_dir in self.training_data_dirs:
                if not os.path.exists(data_dir):
                    raise FileNotFoundError(f"Training data directory not found: {data_dir}")
        
        # Validate data type consistency
        if self.run_game_collection and not self.raw_trmph_data_dirs:
            raise ValueError("Game collection enabled but no raw TRMPH data directories specified. Use --raw-trmph-data-dirs")
        
        if self.run_preprocessing and not self.cleaned_trmph_data_dirs and not self.selfplay_dir:
            raise ValueError("Preprocessing enabled but no input data specified. Use --cleaned-trmph-data-dirs or provide selfplay data")
        
        # Validate ordered positions directories
        if self.ordered_positions_dirs and len(self.ordered_positions_dirs) > 1:
            raise NotImplementedError(
                f"Multiple ordered positions directories not yet supported. "
                f"Found {len(self.ordered_positions_dirs)} directories: {self.ordered_positions_dirs}. "
                f"Please specify only one directory or implement multi-directory support."
            )
        
        # Validate shard ranges are provided when training data directories are specified
        if self.run_training and self.training_data_dirs:
            if self.shard_ranges is None:
                raise ValueError(
                    f"--shard-ranges is required when --training-data-dirs is provided.\n"
                    f"Found {len(self.training_data_dirs)} training data directories but no shard ranges.\n"
                    f"Provide --shard-ranges with {len(self.training_data_dirs)} range(s) (one per directory)."
                )
            if len(self.shard_ranges) != len(self.training_data_dirs):
                raise ValueError(
                    f"Number of shard ranges ({len(self.shard_ranges)}) must match number of training data directories ({len(self.training_data_dirs)}).\n"
                    f"Training data dirs: {self.training_data_dirs}\n"
                    f"Shard ranges: {self.shard_ranges}"
                )
        
        # Resolve validation configuration
        resolved_validation_dirs, resolved_validation_ranges = resolve_validation_config(
            validation_dirs=self.validation_dirs,
            validation_shard_ranges=self.validation_shard_ranges,
            no_validation=self.no_validation
        )
        
        # Store resolved validation configuration
        self.resolved_validation_dirs = resolved_validation_dirs
        self.resolved_validation_ranges = resolved_validation_ranges
        
        # Note: ordered positions directory will be resolved at runtime
        
        # Validate that collected data will be used for training
        if self.run_game_collection and not self.run_preprocessing and not self.run_trmph_processing and not self.run_shuffling:
            raise ValueError(
                "Game collection enabled but all processing steps are disabled. "
                "The collected data will not be used for training. "
                "Either enable preprocessing steps or use --cleaned-trmph-data-dirs for already processed data."
            )

        if self.restart_every_mini_epochs < 0:
            raise ValueError(
                f"restart_every_mini_epochs must be >= 0, got {self.restart_every_mini_epochs}"
            )

        if self.policy_provenance_mode not in {"off", "require"}:
            raise ValueError(
                f"policy_provenance_mode must be 'off' or 'require', got {self.policy_provenance_mode!r}"
            )
    
    def _resolve_ordered_positions_dir(self, newly_created_dir: Optional[str] = None) -> Optional[str]:
        """Resolve the ordered positions directory to use (single source of truth)."""
        # Priority: newly created > existing specified > None
        if newly_created_dir:
            return newly_created_dir
        elif self.ordered_positions_dirs:
            assert len(self.ordered_positions_dirs) == 1, "Only one ordered positions directory is supported"
            return self.ordered_positions_dirs[0]  # Safe because we validated only one exists
        else:
            return None


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
        
        # Use raw TRMPH data directories for collection
        source_dirs = [Path(d) for d in self.config.raw_trmph_data_dirs]
        output_dir = Path(self.config.selfplay_dir)
        
        if not source_dirs:
            raise ValueError("No raw TRMPH data directories specified for game collection. Use --raw-trmph-data-dirs")
        
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
                batch_size=self.config.batch_size,
                cache_size=self.config.cache_size,
                temperature=self.config.temperature,
                verbose=1,
                streaming_save=True,
                write_provenance=self.config.write_provenance,
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
    
    def run(self, input_dir: Optional[str] = None) -> str:
        """Preprocess data and return cleaned directory."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: DATA PREPROCESSING")
        self.logger.info("=" * 60)
        
        # Determine input sources
        input_sources = []
        
        # If we have cleaned TRMPH data directories, only process those (don't mix with selfplay data)
        if self.config.cleaned_trmph_data_dirs:
            for cleaned_dir in self.config.cleaned_trmph_data_dirs:
                input_sources.append(Path(cleaned_dir))
        else:
            # Only use selfplay input if we don't have cleaned data directories
            if input_dir:
                input_sources.append(Path(input_dir))
        
        if not input_sources:
            raise ValueError("No input data specified for preprocessing. Provide either selfplay data or use --cleaned-trmph-data-dirs")
        
        self.logger.info(f"Input sources: {[str(d) for d in input_sources]}")
        self.logger.info(f"Output directory: {self.config.cleaned_dir}")
        self.logger.info(f"Chunk size: {self.config.chunk_size}")
        
        # Check if output already exists
        if Path(self.config.cleaned_dir).exists() and list(Path(self.config.cleaned_dir).glob("*.trmph")):
            raise FileExistsError(
                f"Output directory already exists and contains data: {self.config.cleaned_dir}\n"
                f"This suggests the data has already been processed. To avoid wasting compute time,\n"
                f"either:\n"
                f"1. Use a different output directory\n"
                f"2. Remove the existing output directory\n"
                f"3. Use --no-preprocessing to skip this step"
            )
        
        # Create output directory
        Path(self.config.cleaned_dir).mkdir(parents=True, exist_ok=True)
        
        # Check that all input sources exist - fail fast if any are missing
        missing_sources = [source_dir for source_dir in input_sources if not source_dir.exists()]
        if missing_sources:
            raise FileNotFoundError(f"Input sources do not exist: {missing_sources}. This is likely a configuration error.")
        
        # Process all sources together
        self.logger.info(f"Processing {len(input_sources)} input sources together")
        combine_and_clean_files(
            input_sources,
            Path(self.config.cleaned_dir),
            self.config.chunk_size,
            policy_provenance_mode=self.config.policy_provenance_mode,
        )
        
        # Verify output was created
        output_files = list(Path(self.config.cleaned_dir).glob("*.trmph"))
        if not output_files:
            raise RuntimeError(f"Preprocessing failed: No output files created in {self.config.cleaned_dir}")
        
        self.logger.info(f"Preprocessing completed: {len(output_files)} output files created")
        return self.config.cleaned_dir


class TRMPHProcessingStep:
    """Handles TRMPH file processing into ordered training positions."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run(self, input_dir: str) -> str:
        """Process TRMPH files and return ordered positions directory."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: TRMPH PROCESSING (Creating Ordered Positions)")
        self.logger.info("=" * 60)
        
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {self.config.ordered_positions_dir}")
        self.logger.info(f"Position selector: {self.config.position_selector}")
        self.logger.info(f"Policy provenance mode: {self.config.policy_provenance_mode}")
        self.logger.info(f"Max workers: {self.config.max_workers_trmph}")
        
        # Check if output already exists
        if Path(self.config.ordered_positions_dir).exists() and list(Path(self.config.ordered_positions_dir).glob("*.pkl.gz")):
            raise FileExistsError(
                f"Output directory already exists and contains data: {self.config.ordered_positions_dir}\n"
                f"This suggests the data has already been processed. To avoid wasting compute time,\n"
                f"either:\n"
                f"1. Use a different --selfplay-dir\n"
                f"2. Remove the existing output directory\n"
                f"3. Use --no-trmph-processing to skip this step"
            )
        
        # Create output directory
        Path(self.config.ordered_positions_dir).mkdir(parents=True, exist_ok=True)
        
        # Create configuration
        config = create_config_from_args(type('Args', (), {
            'data_dir': input_dir,
            'output_dir': self.config.ordered_positions_dir,
            'max_files': None,
            'position_selector': self.config.position_selector,
            'policy_provenance_mode': self.config.policy_provenance_mode,
            'run_tag': f"pipeline_{self.config.run_timestamp}",
            'max_workers': self.config.max_workers_trmph,
            'sequential': False
        })())
        
        # Process files
        results = process_files(config)
        
        # Verify output was created
        output_files = list(Path(self.config.ordered_positions_dir).glob("*.pkl.gz"))
        if not output_files:
            raise RuntimeError(f"TRMPH processing failed: No output files created in {self.config.ordered_positions_dir}")
        
        self.logger.info(f"TRMPH processing completed: {len(output_files)} output files created")
        self.logger.info(f"Results: {results}")
        
        return self.config.ordered_positions_dir


class ShufflingStep:
    """Handles shuffling of ordered positions into final training data."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run(self, input_dir: str) -> str:
        """Shuffle ordered positions and return shuffled training data directory."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: DATA SHUFFLING (Creating Final Training Data)")
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
        
        # Verify output was created
        output_files = list(Path(self.config.shuffled_dir).glob("*.pkl.gz"))
        if not output_files:
            raise RuntimeError(f"Shuffling failed: No output files created in {self.config.shuffled_dir}")
        
        self.logger.info(f"Shuffling completed: {len(output_files)} output files created")
        
        return self.config.shuffled_dir


class TrainingStep:
    """Handles model training."""

    NUM_EPOCHS_PER_PIPELINE_RUN = 4
    MINI_EPOCH_SAMPLES = 250000
    TRAINING_RANDOM_SEED = 42
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _build_experiments(self) -> List[Dict[str, Any]]:
        """Build experiment definitions from the hyperparameter sweep config."""
        sweep = create_hyperparameter_sweep(self.config.hyperparameter_overrides)

        import itertools
        param_names = list(sweep.keys())
        param_values = list(sweep.values())
        all_configs = list(itertools.product(*param_values))

        experiments: List[Dict[str, Any]] = []
        for i, config_values in enumerate(all_configs):
            config = dict(zip(param_names, config_values))

            # Keep policy/value loss weights normalized.
            if "policy_weight" in config:
                config["value_weight"] = 1.0 - config["policy_weight"]

            exp_name = f"pipeline_sweep_{i}"
            if len(all_configs) > 1:
                varying_params = [k for k, v in sweep.items() if len(v) > 1]
                if varying_params:
                    labels = []
                    for param in varying_params:
                        short_label = HYPERPARAMETER_SHORT_LABELS.get(param, param)
                        value = config[param]
                        if isinstance(value, float):
                            labels.append(f"{short_label}{value:.0e}")
                        else:
                            labels.append(f"{short_label}{value}")
                    exp_name = f"pipeline_sweep_{i}_{'_'.join(labels)}"

            experiments.append(
                {
                    "experiment_name": exp_name,
                    "hyperparameters": config,
                }
            )
        return experiments

    def _resolve_training_data_sources(self, new_shuffled_dir: Optional[str]):
        if new_shuffled_dir:
            all_data_dirs = [new_shuffled_dir] + self.config.training_data_dirs
            all_shard_ranges = ["all"] + self.config.shard_ranges
        else:
            all_data_dirs = self.config.training_data_dirs
            all_shard_ranges = self.config.shard_ranges

        return (
            all_data_dirs,
            all_shard_ranges,
            self.config.resolved_validation_dirs,
            self.config.resolved_validation_ranges,
        )

    @staticmethod
    def _parse_epoch_mini_from_checkpoint(checkpoint_path: Path) -> tuple[int, int]:
        match = re.search(r"epoch(\d+)_mini(\d+)", checkpoint_path.name)
        if not match:
            raise ValueError(
                f"Could not parse epoch/mini from checkpoint filename: {checkpoint_path.name}"
            )
        return int(match.group(1)), int(match.group(2))

    def _find_latest_checkpoint(self, checkpoint_dir: Path) -> Path:
        candidates = list(checkpoint_dir.glob("epoch*_mini*.pt*"))
        if not candidates:
            raise FileNotFoundError(
                f"No epoch checkpoint files found in {checkpoint_dir}"
            )

        def _key(path: Path):
            epoch, mini = self._parse_epoch_mini_from_checkpoint(path)
            return (epoch, mini)

        return max(candidates, key=_key)

    def _run_training_once(
        self,
        *,
        experiments: List[Dict[str, Any]],
        all_data_dirs: List[str],
        all_shard_ranges: List[str],
        all_validation_dirs: List[str],
        all_validation_shard_ranges: List[str],
        results_dir: str,
        resume_from: str,
        max_mini_epochs: Optional[int],
        resume_mode: str,
        target_end_epoch: Optional[int],
        allow_missing_stream_sidecar_fallback: bool,
    ) -> Dict[str, Any]:
        shutdown_handler = GracefulShutdown()
        return run_hyperparameter_tuning_current_data(
            experiments=experiments,
            data_dirs=all_data_dirs,
            validation_dirs=all_validation_dirs,
            validation_shard_ranges=all_validation_shard_ranges,
            results_dir=results_dir,
            train_ratio=0.8,
            num_epochs=self.NUM_EPOCHS_PER_PIPELINE_RUN,
            early_stopping_patience=None,
            random_seed=self.TRAINING_RANDOM_SEED,
            max_examples_unaugmented=self.config.max_samples,
            max_validation_examples=self.config.max_validation_samples,
            experiment_name=None,
            enable_augmentation=True,
            mini_epoch_samples=self.MINI_EPOCH_SAMPLES,
            resume_from=resume_from,
            shard_ranges=all_shard_ranges,
            shutdown_handler=shutdown_handler,
            run_timestamp=self.config.run_timestamp,
            override_checkpoint_hyperparameters=self.config.override_checkpoint_hyperparameters,
            shuffle_shards=True,
            max_mini_epochs=max_mini_epochs,
            resume_mode=resume_mode,
            target_end_epoch=target_end_epoch,
            allow_missing_stream_sidecar_fallback=allow_missing_stream_sidecar_fallback,
        )

    def _build_child_chunk_command(
        self,
        *,
        checkpoint_path: Path,
        all_data_dirs: List[str],
        all_shard_ranges: List[str],
        all_validation_dirs: List[str],
        all_validation_shard_ranges: List[str],
        resume_mode: str,
        target_end_epoch: int,
    ) -> List[str]:
        model_epoch, model_mini = self._parse_epoch_mini_from_checkpoint(checkpoint_path)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model-path",
            str(checkpoint_path.parent),
            "--model-epoch",
            str(model_epoch),
            "--model-mini",
            str(model_mini),
            "--no-selfplay",
            "--no-preprocessing",
            "--no-trmph-processing",
            "--no-shuffling",
            "--no-cleanup",
            "--training-data-dirs",
            *all_data_dirs,
            "--shard-ranges",
            *all_shard_ranges,
            "--results-dir",
            self.config.results_dir,
            "--max-samples",
            str(self.config.max_samples),
            "--max-validation-samples",
            str(self.config.max_validation_samples),
            "--restart-every-mini-epochs",
            str(self.config.restart_every_mini_epochs),
            "--run-timestamp",
            self.config.run_timestamp,
            "--internal-training-chunk-run",
            "--max-mini-epochs-per-run",
            str(self.config.restart_every_mini_epochs),
            "--resume-mode",
            resume_mode,
            "--target-end-epoch",
            str(target_end_epoch),
        ]

        if all_validation_dirs and all_validation_shard_ranges:
            cmd.extend(["--validation-dirs", *all_validation_dirs])
            cmd.extend(["--validation-shard-ranges", *all_validation_shard_ranges])
        else:
            cmd.append("--no-validation")

        if self.config.override_checkpoint_hyperparameters:
            cmd.append("--override-checkpoint-hyperparameters")
        if self.config.allow_missing_stream_sidecar_fallback:
            cmd.append("--allow-missing-stream-sidecar-fallback")

        # Preserve explicit hyperparameter overrides across chunk runs.
        override_arg_map = {
            "learning_rate": "--learning-rate",
            "batch_size": "--train-batch-size",
            "weight_decay": "--weight-decay",
            "policy_weight": "--policy-weight",
            "max_grad_norm": "--max-grad-norm",
            "value_learning_rate_factor": "--value-learning-rate-factor",
            "value_weight_decay_factor": "--value-weight-decay-factor",
        }
        for key, arg_name in override_arg_map.items():
            values = self.config.hyperparameter_overrides.get(key)
            if values:
                cmd.extend([arg_name, str(values[0])])

        return cmd

    def _run_training_with_restarts(
        self,
        *,
        experiments: List[Dict[str, Any]],
        all_data_dirs: List[str],
        all_shard_ranges: List[str],
        all_validation_dirs: List[str],
        all_validation_shard_ranges: List[str],
        results_dir: str,
    ) -> str:
        if len(experiments) != 1:
            self.logger.warning(
                "Automatic chunked restarts currently support a single experiment. "
                "Falling back to a single-process training run."
            )
            results = self._run_training_once(
                experiments=experiments,
                all_data_dirs=all_data_dirs,
                all_shard_ranges=all_shard_ranges,
                all_validation_dirs=all_validation_dirs,
                all_validation_shard_ranges=all_validation_shard_ranges,
                results_dir=results_dir,
                resume_from=self.config.model_full_path,
                max_mini_epochs=None,
                resume_mode="next_epoch",
                target_end_epoch=self.config.target_end_epoch,
                allow_missing_stream_sidecar_fallback=self.config.allow_missing_stream_sidecar_fallback,
            )
            self.logger.info(f"Training completed: {results}")
            return results_dir

        starting_checkpoint = Path(self.config.model_full_path)
        if not starting_checkpoint.exists():
            raise FileNotFoundError(
                f"Starting checkpoint not found: {starting_checkpoint}"
            )
        initial_epoch, _initial_mini = self._parse_epoch_mini_from_checkpoint(starting_checkpoint)
        target_end_epoch = (
            self.config.target_end_epoch
            if self.config.target_end_epoch is not None
            else (initial_epoch + self.NUM_EPOCHS_PER_PIPELINE_RUN)
        )

        self.logger.info(
            f"Chunked training restart mode enabled (every {self.config.restart_every_mini_epochs} mini-epochs). "
            f"Target end epoch: {target_end_epoch}"
        )

        latest_resume_checkpoint = starting_checkpoint
        chunk_idx = 0

        while True:
            chunk_idx += 1
            resume_mode = "same_epoch"
            child_cmd = self._build_child_chunk_command(
                checkpoint_path=latest_resume_checkpoint,
                all_data_dirs=all_data_dirs,
                all_shard_ranges=all_shard_ranges,
                all_validation_dirs=all_validation_dirs,
                all_validation_shard_ranges=all_validation_shard_ranges,
                resume_mode=resume_mode,
                target_end_epoch=target_end_epoch,
            )

            epoch, mini = self._parse_epoch_mini_from_checkpoint(latest_resume_checkpoint)
            self.logger.info(
                f"Starting training chunk {chunk_idx}: resume checkpoint epoch{epoch}_mini{mini}, "
                f"resume_mode={resume_mode}"
            )
            child_result = subprocess.run(child_cmd, check=False)
            if child_result.returncode != 0:
                raise RuntimeError(
                    f"Training chunk {chunk_idx} failed with exit code {child_result.returncode}"
                )

            overall_results_path = Path(results_dir) / "overall_results.json"
            if not overall_results_path.exists():
                raise RuntimeError(
                    f"Expected chunk results file not found: {overall_results_path}"
                )
            with open(overall_results_path, "r", encoding="utf-8") as f:
                chunk_results = json.load(f)

            chunk_experiments = chunk_results.get("experiments")
            if not isinstance(chunk_experiments, list) or not chunk_experiments:
                raise RuntimeError(
                    f"Invalid chunk results format in {overall_results_path}: missing experiments list"
                )
            chunk_result = chunk_experiments[0]

            if chunk_result.get("already_complete"):
                self.logger.info("Chunked training supervisor: target already reached.")
                break

            stopped_due_to_cap = bool(chunk_result.get("stopped_due_to_max_mini_epochs"))
            if not stopped_due_to_cap:
                self.logger.info("Chunked training supervisor: training completed all planned epochs.")
                break

            next_checkpoint = self._find_latest_checkpoint(Path(results_dir))
            if next_checkpoint == latest_resume_checkpoint:
                raise RuntimeError(
                    f"Chunk {chunk_idx} completed but did not produce a newer checkpoint in {results_dir}"
                )
            latest_resume_checkpoint = next_checkpoint

        return results_dir

    def run(self, new_shuffled_dir: str):
        """Run training with the new data."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: MODEL TRAINING")
        self.logger.info("=" * 60)

        results_dir = str(Path(self.config.results_dir) / f"pipeline_{self.config.run_timestamp}")
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info(f"New training data directory: {new_shuffled_dir}")
        self.logger.info(f"Existing training data directories: {self.config.training_data_dirs}")
        self.logger.info(f"Shard ranges: {self.config.shard_ranges}")
        self.logger.info(f"Results directory: {results_dir}")
        self.logger.info(f"Max samples: {self.config.max_samples}")
        self.logger.info(f"Resume from: {self.config.model_full_path}")

        experiments = self._build_experiments()

        if self.config.enable_memory_profiling:
            take_snapshot("training_start")

        (
            all_data_dirs,
            all_shard_ranges,
            all_validation_dirs,
            all_validation_shard_ranges,
        ) = self._resolve_training_data_sources(new_shuffled_dir)

        if self.config.restart_every_mini_epochs > 0 and not self.config.internal_training_chunk_run:
            return self._run_training_with_restarts(
                experiments=experiments,
                all_data_dirs=all_data_dirs,
                all_shard_ranges=all_shard_ranges,
                all_validation_dirs=all_validation_dirs,
                all_validation_shard_ranges=all_validation_shard_ranges,
                results_dir=results_dir,
            )

        results = self._run_training_once(
            experiments=experiments,
            all_data_dirs=all_data_dirs,
            all_shard_ranges=all_shard_ranges,
            all_validation_dirs=all_validation_dirs,
            all_validation_shard_ranges=all_validation_shard_ranges,
            results_dir=results_dir,
            resume_from=self.config.model_full_path,
            max_mini_epochs=self.config.max_mini_epochs_per_run,
            resume_mode=self.config.resume_mode,
            target_end_epoch=self.config.target_end_epoch,
            allow_missing_stream_sidecar_fallback=self.config.allow_missing_stream_sidecar_fallback,
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
        self.logger.info("=" * 60)
        self.logger.info("HEX AI TRAINING PIPELINE")
        self.logger.info("=" * 60)
        self.logger.info(f"Run timestamp: {self.config.run_timestamp}")
        self.logger.info(f"Model: {self.config.model_full_path}")
        self.logger.info(f"Configuration: {self.config}")
        
        # Validate configuration
        self.config.validate(
            check_model=self.config.run_selfplay or self.config.run_training,
            check_data=self.config.run_training
        )
        
        start_time = time.time()
        
        # Start memory profiling if enabled
        if self.config.enable_memory_profiling:
            start_profiling(interval_seconds=self.config.memory_profile_interval_seconds)
            take_snapshot("pipeline_start")
        
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
                if selfplay_dir and not self.config.run_preprocessing:
                    self.logger.warning("WARNING: Self-play data available but preprocessing is disabled. The self-play data will not be used for training.")
                self.logger.info("Skipping preprocessing (disabled or no self-play data)")
                cleaned_dir = None
            
            # Step 3: TRMPH processing (optional)
            if self.config.run_trmph_processing and cleaned_dir:
                self.current_step = 3
                self.logger.info(f"\nStarting step {self.current_step}: TRMPH processing (creating ordered positions)")
                ordered_positions_dir = self.trmph_step.run(cleaned_dir)
                self.step_results['trmph_processing'] = ordered_positions_dir
            else:
                if cleaned_dir and not self.config.run_trmph_processing:
                    self.logger.warning("WARNING: Preprocessing completed but TRMPH processing is disabled. The cleaned data will not be converted to ordered positions.")
                self.logger.info("Skipping TRMPH processing (disabled or no cleaned data)")
                ordered_positions_dir = None
            
            # Step 4: Shuffling (optional)
            # Resolve ordered positions directory (single source of truth)
            resolved_ordered_positions_dir = self.config._resolve_ordered_positions_dir(ordered_positions_dir)
            
            if self.config.run_shuffling and resolved_ordered_positions_dir:
                self.current_step = 4
                self.logger.info(f"\nStarting step {self.current_step}: Shuffling (creating final training data)")
                self.logger.info(f"Using ordered positions: {resolved_ordered_positions_dir}")
                
                shuffled_dir = self.shuffling_step.run(resolved_ordered_positions_dir)
                self.step_results['shuffling'] = shuffled_dir
            else:
                if resolved_ordered_positions_dir and not self.config.run_shuffling:
                    self.logger.warning("WARNING: Ordered positions available but shuffling is disabled. The ordered positions will not be shuffled for training.")
                    # raise ValueError("Ordered positions available but shuffling is disabled. The ordered positions will not be shuffled for training.")
                self.logger.info("Skipping shuffling (disabled or no ordered positions data)")
                shuffled_dir = None
            
            # Step 5: Training (optional)
            if self.config.run_training:
                self.current_step = 5
                self.logger.info(f"\nStarting step {self.current_step}: Training")
                
                # Use newly shuffled data if available, otherwise use existing training data sources
                if shuffled_dir:
                    training_data_dir = shuffled_dir
                    self.logger.info(f"Using newly shuffled training data: {training_data_dir}")
                elif self.config.training_data_dirs:
                    # When no new shuffled data, just use existing training data (no duplication)
                    training_data_dir = None
                    self.logger.info(f"Using existing training data: {self.config.training_data_dirs}")
                else:
                    self.logger.error("No training data available")
                    raise ValueError("No training data available - need either shuffled data or training data sources")
                
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
            
            # Stop memory profiling if enabled
            if self.config.enable_memory_profiling:
                take_snapshot("pipeline_end")
                stop_profiling()
            
        except GracefulShutdownRequested:
            self.logger.info("Pipeline interrupted by graceful shutdown request")
            raise
        except Exception as e:
            self.logger.error(f"Pipeline failed at step {self.current_step}: {e}")
            self.logger.error("Step results so far:")
            for step, result in self.step_results.items():
                self.logger.error(f"  {step}: {result}")
            # Stop memory profiling if enabled (even on error)
            if self.config.enable_memory_profiling:
                take_snapshot("pipeline_error")
                stop_profiling()
            raise
    
    def _cleanup_intermediate_files(self):
        """Clean up intermediate files to save disk space."""
        self.logger.info("Cleaning up intermediate files...")
        
        # Remove temporary directories
        temp_dirs = [
            self.config.selfplay_dir,
            self.config.cleaned_dir,
            self.config.ordered_positions_dir,  # Clean up ordered positions (intermediate)
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
  
  # Collect raw data and train (tournament_play and sf25 included by default)
  python scripts/training_pipeline.py --use-current-best-model --run-game-collection --no-selfplay
  
  # Collect from specific directories only
  python scripts/training_pipeline.py --use-current-best-model --raw-trmph-data-dirs data/sf25/sep8 --run-game-collection --no-selfplay
  
  # Use existing cleaned data
  python scripts/training_pipeline.py --use-current-best-model --cleaned-trmph-data-dirs data/collected/tournament_sep3_8 data/collected/sep8_games --no-selfplay --no-preprocessing --no-trmph-processing --no-shuffling
  
  # Mix all data types
  python scripts/training_pipeline.py --use-current-best-model --raw-trmph-data-dirs data/sf25/sep8 --cleaned-trmph-data-dirs data/collected/tournament_sep3_8 --training-data-dirs data/processed/sf18_shuffled --shard-ranges "221-250" --run-game-collection --no-selfplay

  # Train only on sf18_shuffled (no self-play or preprocessing)
  python scripts/training_pipeline.py --use-current-best-model --training-data-dirs data/processed/sf18_shuffled --shard-ranges "all" --no-selfplay --no-preprocessing --no-trmph-processing --no-shuffling
  
  # Run only self-play and preprocessing
  python scripts/training_pipeline.py --use-current-best-model --no-training --no-shuffling --no-trmph-processing
  
  # Run with custom settings
  python scripts/training_pipeline.py --use-current-best-model --num-games 50000 --num-workers 5 --temperature 1.0
  
  # Use existing raw self-play data
  python scripts/training_pipeline.py --use-current-best-model --selfplay-dir data/sf25/aug_04 --no-selfplay
        """
    )
    
    # Model configuration
    parser.add_argument("--model-path", help="Path to model checkpoint directory")
    parser.add_argument("--model-epoch", type=int, help="Model epoch number (required unless using --use-current-best-model)")
    parser.add_argument("--model-mini", type=int, help="Model mini-epoch number (required unless using --use-current-best-model)")
    parser.add_argument("--use-current-best-model", action="store_true", 
                       help="Use current best model from hex_ai.inference.model_config")
    
    # Self-play configuration
    parser.add_argument("--num-games", type=int, default=100000, help="Number of games to generate")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of self-play workers")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_START, help=f"Temperature for move sampling (default: {DEFAULT_TEMPERATURE_START})")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--cache-size", type=int, default=DEFAULT_CACHE_SIZE, help=f"Cache size for model inference (default: {DEFAULT_CACHE_SIZE})")
    parser.add_argument(
        "--no-write-provenance",
        action="store_true",
        help="Disable move-provenance sidecar writing during self-play generation.",
    )
    
    # Data configuration
    parser.add_argument("--base-data-dir", default="data", help="Base directory for data")
    parser.add_argument("--selfplay-dir", help="Use existing raw self-play directory (skip self-play generation)")
    
    # Explicit data type arguments
    parser.add_argument("--raw-trmph-data-dirs", type=str, nargs='+', 
                       default=[str(d) for d in hex_ai.data_config.DEFAULT_SOURCE_DIRS],
                       help="Raw .trmph files to collect and clean (for game collection step). Defaults to tournament_play and sf25 directories.")
    parser.add_argument("--cleaned-trmph-data-dirs", type=str, nargs='+', default=[],
                       help="Already cleaned .trmph files to process (for preprocessing step)")
    parser.add_argument("--ordered-positions-dirs", type=str, nargs='+', default=[],
                       help="Existing ordered positions directories (not shuffled) for shuffling step")
    parser.add_argument("--training-data-dirs", type=str, nargs='+', 
                       default=[str(d) for d in hex_ai.data_config.DEFAULT_TRAINING_DATA_DIRS],
                       help="Existing training data directories (shuffled positions) for training")
    parser.add_argument("--shard-ranges", type=str, nargs='+',
                       help='Shard ranges for training data directories. Format: "start-end", comma-separated ranges like "0-206,208-498", or "all" (e.g., --shard-ranges "251-300" "all").')
    # Validation data arguments
    validation_group = parser.add_argument_group('validation data')
    
    validation_group.add_argument(
        '--validation-dirs',
        type=str,
        nargs='*',
        help='Validation data directories (defaults to hardcoded values)'
    )
    
    validation_group.add_argument(
        '--validation-shard-ranges',
        type=str,
        nargs='*',
        help='Validation shard ranges (defaults to hardcoded values)'
    )
    
    validation_group.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable validation entirely'
    )
    
    parser.add_argument("--chunk-size", type=int, default=10000, help="Chunk size for preprocessing")
    parser.add_argument("--position-selector", default="all", choices=["all", "final", "penultimate"], help="Position selector for TRMPH processing")
    parser.add_argument(
        "--policy-provenance-mode",
        default="require",
        choices=["off", "require"],
        help=(
            "Policy provenance handling during TRMPH processing: "
            "'off' ignores sidecars, 'require' enforces sidecar alignment and masks policy targets."
        ),
    )
    parser.add_argument("--max-workers-trmph", type=int, default=6, help="Max workers for TRMPH processing")
    parser.add_argument("--num-buckets-shuffle", type=int, default=100, help="Number of buckets for shuffling")
    
    # Training configuration
    parser.add_argument("--max-samples", type=int, default=35000000, help="Max training samples")
    parser.add_argument("--max-validation-samples", type=int, default=50000, help="Max validation samples")
    parser.add_argument("--results-dir", default="checkpoints/hyperparameter_tuning", help="Results directory")
    parser.add_argument(
        "--restart-every-mini-epochs",
        type=int,
        default=10,
        help="Restart the training subprocess every N mini-epochs (default: 10, set 0 to disable).",
    )
    parser.add_argument(
        "--allow-missing-stream-sidecar-fallback",
        action="store_true",
        help=(
            "Allow same-epoch resume to fall back when stream-state sidecar is missing/unavailable. "
            "Disabled by default to fail fast on restart-state inconsistencies."
        ),
    )
    parser.add_argument("--override-checkpoint-hyperparameters", action="store_true", 
                       help="Override checkpoint hyperparameters with current sweep settings (resets optimizer state)")
    
    # Hyperparameter override arguments
    parser.add_argument("--learning-rate", type=float, help="Override learning rate (e.g., 1e-4)")
    parser.add_argument("--train-batch-size", type=int, help="Override training batch size")
    parser.add_argument("--weight-decay", type=float, help="Override weight decay")
    parser.add_argument("--policy-weight", type=float, help="Override policy weight (value weight will be 1-policy_weight)")
    parser.add_argument("--max-grad-norm", type=float, help="Override max gradient norm")
    parser.add_argument("--value-learning-rate-factor", type=float, help="Override value learning rate factor")
    parser.add_argument("--value-weight-decay-factor", type=float, help="Override value weight decay factor")
    
    # Pipeline control
    parser.add_argument("--run-game-collection", action="store_true", help="Run game collection from multiple sources")
    parser.add_argument("--no-selfplay", action="store_true", help="Skip self-play step")
    parser.add_argument("--no-preprocessing", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--no-trmph-processing", action="store_true", help="Skip TRMPH processing step")
    parser.add_argument("--no-shuffling", action="store_true", help="Skip shuffling step")
    parser.add_argument("--no-training", action="store_true", help="Skip training step")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep intermediate files")
    parser.add_argument("--enable-memory-profiling", action="store_true", 
                       help="Enable memory profiling (tracks RSS vs heap and takes snapshots)")
    parser.add_argument("--memory-profile-interval-seconds", type=int, default=60,
                       help="Sampling interval for memory profiling timeline (default: 60)")

    # Internal chunk-run controls (hidden)
    parser.add_argument("--internal-training-chunk-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-mini-epochs-per-run", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume-mode",
        choices=["next_epoch", "same_epoch"],
        default="next_epoch",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--target-end-epoch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--run-timestamp", type=str, help=argparse.SUPPRESS)
    
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
                model_path = get_model_path("best")
                args.model_path = get_model_dir("best")
                
                # Extract epoch and mini from the filename
                import os
                import re
                filename = os.path.basename(model_path)
                # Expected format: epoch2_mini201.pt.gz
                # Use regex to extract epoch and mini numbers
                match = re.search(r'epoch(\d+)_mini(\d+)\.pt\.gz', filename)
                if not match:
                    raise ValueError(
                        f"Could not extract epoch and mini from model filename: {filename}\n"
                        f"Expected format: epoch<number>_mini<number>.pt.gz\n"
                        f"Full model path: {model_path}"
                    )
                args.model_epoch = int(match.group(1))
                args.model_mini = int(match.group(2))
                
                logger.info(f"Using current best model: {model_path}")
                logger.info(f"Model directory: {args.model_path}")
                logger.info(f"Model epoch: {args.model_epoch}, mini: {args.model_mini}")
            except ImportError:
                raise ValueError("Could not import hex_ai.inference.model_config")
            except Exception as e:
                raise ValueError(f"Could not get current best model path: {e}")
        elif not args.model_path:
            raise ValueError("Must specify either --model-path or --use-current-best-model")
        
        # Validate that model_epoch and model_mini are set when using --model-path
        if not args.use_current_best_model:
            if args.model_epoch is None:
                raise ValueError("--model-epoch is required when using --model-path")
            if args.model_mini is None:
                raise ValueError("--model-mini is required when using --model-path")
        

        # Collect hyperparameter overrides
        hyperparameter_overrides = {}
        if args.learning_rate is not None:
            hyperparameter_overrides["learning_rate"] = [args.learning_rate]
        if args.train_batch_size is not None:
            hyperparameter_overrides["batch_size"] = [args.train_batch_size]
        if args.weight_decay is not None:
            hyperparameter_overrides["weight_decay"] = [args.weight_decay]
        if args.policy_weight is not None:
            hyperparameter_overrides["policy_weight"] = [args.policy_weight]
        if args.max_grad_norm is not None:
            hyperparameter_overrides["max_grad_norm"] = [args.max_grad_norm]
        if args.value_learning_rate_factor is not None:
            hyperparameter_overrides["value_learning_rate_factor"] = [args.value_learning_rate_factor]
        if args.value_weight_decay_factor is not None:
            hyperparameter_overrides["value_weight_decay_factor"] = [args.value_weight_decay_factor]

        # Create configuration
        config = PipelineConfig(
            model_path=args.model_path,
            model_epoch=args.model_epoch,
            model_mini=args.model_mini,
            num_games=args.num_games,
            num_workers=args.num_workers,
            temperature=args.temperature,
            batch_size=args.batch_size,
            cache_size=args.cache_size,
            write_provenance=not args.no_write_provenance,
            base_data_dir=args.base_data_dir,
            raw_trmph_data_dirs=args.raw_trmph_data_dirs,
            cleaned_trmph_data_dirs=args.cleaned_trmph_data_dirs,
            ordered_positions_dirs=args.ordered_positions_dirs,
            training_data_dirs=args.training_data_dirs,
            shard_ranges=args.shard_ranges,
            validation_dirs=args.validation_dirs,
            validation_shard_ranges=args.validation_shard_ranges,
            no_validation=args.no_validation,
            selfplay_dir=args.selfplay_dir,
            chunk_size=args.chunk_size,
            position_selector=args.position_selector,
            policy_provenance_mode=args.policy_provenance_mode,
            max_workers_trmph=args.max_workers_trmph,
            num_buckets_shuffle=args.num_buckets_shuffle,
            max_samples=args.max_samples,
            max_validation_samples=args.max_validation_samples,
            results_dir=args.results_dir,
            restart_every_mini_epochs=args.restart_every_mini_epochs,
            allow_missing_stream_sidecar_fallback=args.allow_missing_stream_sidecar_fallback,
            max_mini_epochs_per_run=args.max_mini_epochs_per_run,
            resume_mode=args.resume_mode,
            target_end_epoch=args.target_end_epoch,
            internal_training_chunk_run=args.internal_training_chunk_run,
            run_timestamp_override=args.run_timestamp,
            override_checkpoint_hyperparameters=args.override_checkpoint_hyperparameters,
            hyperparameter_overrides=hyperparameter_overrides,
            run_game_collection=args.run_game_collection,
            run_selfplay=not args.no_selfplay and args.selfplay_dir is None,
            run_preprocessing=not args.no_preprocessing,
            run_trmph_processing=not args.no_trmph_processing,
            run_shuffling=not args.no_shuffling,
            run_training=not args.no_training,
            cleanup_intermediate=not args.no_cleanup,
            enable_memory_profiling=args.enable_memory_profiling,
            memory_profile_interval_seconds=args.memory_profile_interval_seconds
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
