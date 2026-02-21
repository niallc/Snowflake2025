import logging
import sys
import numpy as np
from hex_ai.error_handling import GracefulShutdownRequested
from hex_ai.memory_profiler import get_profiler, write_epoch_summary
from hex_ai.memory_leak_diagnostics import get_diagnostics

class MiniEpochOrchestrator:
    """
    Orchestrates training in mini-epochs, enabling validation and checkpointing at configurable sample intervals.

    This class wraps a Trainer and DataLoader, running training in mini-epochs (N samples), and performing
    validation/checkpointing after each mini-epoch. It maintains model/optimizer state across the entire run.

    Args:
        trainer: The Trainer instance (must provide train_on_batches, validate, save_checkpoint).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data (optional).
        mini_epoch_samples: Number of samples per mini-epoch (int).
        num_epochs: Number of full epochs to train.
        checkpoint_dir: Directory to save checkpoints (str or Path).
        log_interval: How often to log progress (in mini-epochs).

    Usage:
        orchestrator = MiniEpochOrchestrator(trainer, train_loader, val_loader, mini_epoch_samples=128000, num_epochs=10)
        orchestrator.run()
    """
    def __init__(self, trainer, train_loader, val_loader=None, mini_epoch_samples=128000, num_epochs=1,
                 checkpoint_dir=None, log_interval=1, shutdown_handler=None, start_epoch=0):
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mini_epoch_samples = mini_epoch_samples
        self.mini_epoch_batches = mini_epoch_samples // train_loader.batch_size
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        self.shutdown_handler = shutdown_handler
        self.start_epoch = start_epoch

    def _log_mini_epoch_memory_markers(self, epoch: int, mini_epoch: int, batch_count: int) -> None:
        """
        Emit lightweight memory/census markers tied to mini-epoch boundaries.
        """
        profiler = get_profiler()
        if profiler is None:
            return

        label = f"epoch_{epoch}_mini_{mini_epoch}_end"
        profiler.log_measurement(label=label)

        metrics = {}
        train_dataset = getattr(self.train_loader, 'dataset', None)
        if train_dataset is not None:
            if hasattr(train_dataset, 'position_pool'):
                position_pool = train_dataset.position_pool
                metrics['train_position_pool_size'] = float(len(position_pool))
                if position_pool:
                    avg_position_bytes = self._estimate_average_entry_bytes(position_pool, max_samples=64)
                    metrics['train_position_avg_bytes'] = float(avg_position_bytes)
                    metrics['train_position_pool_est_mb'] = float(
                        (avg_position_bytes * len(position_pool)) / (1024 ** 2)
                    )
            if hasattr(train_dataset, 'shard_queues'):
                metrics['train_shards_remaining'] = float(
                    sum(len(queue) for queue in train_dataset.shard_queues)
                )
            if hasattr(train_dataset, 'total_positions_yielded'):
                metrics['train_positions_yielded'] = float(train_dataset.total_positions_yielded)

        if self.val_loader is not None:
            val_dataset = getattr(self.val_loader, 'dataset', None)
            if val_dataset is not None:
                if hasattr(val_dataset, 'validation_positions'):
                    metrics['validation_positions_total'] = float(len(val_dataset.validation_positions))
                if hasattr(val_dataset, 'validation_position_index'):
                    metrics['validation_positions_consumed'] = float(val_dataset.validation_position_index)

        if hasattr(self.trainer, 'gradient_clipping_debug'):
            debug_entries = self.trainer.gradient_clipping_debug
            metrics['gradient_clipping_debug_len'] = float(len(debug_entries))
            if debug_entries:
                avg_debug_entry_bytes = self._estimate_average_entry_bytes(debug_entries, max_samples=128)
                metrics['gradient_clipping_debug_avg_bytes'] = float(avg_debug_entry_bytes)
                metrics['gradient_clipping_debug_est_mb'] = float(
                    (avg_debug_entry_bytes * len(debug_entries)) / (1024 ** 2)
                )

        if metrics:
            profiler.log_object_census(metrics, label=label, extra_json={'batch_count': batch_count})

    def _estimate_average_entry_bytes(self, values, max_samples: int = 64) -> float:
        """
        Approximate average deep size for a container's entries.

        This is intentionally heuristic and bounded to keep profiling overhead low.
        """
        if not values:
            return 0.0
        sample_count = min(len(values), max_samples)
        sample = values[:sample_count]
        total = 0
        for item in sample:
            total += self._estimate_object_bytes(item, max_depth=3)
        return total / max(1, sample_count)

    def _estimate_object_bytes(self, obj, max_depth: int = 3, _depth: int = 0, _seen=None) -> int:
        """Best-effort deep size estimate for small Python objects and numpy-like arrays."""
        if _seen is None:
            _seen = set()
        obj_id = id(obj)
        if obj_id in _seen:
            return 0
        _seen.add(obj_id)

        size = sys.getsizeof(obj)
        if isinstance(obj, np.ndarray):
            # For owning arrays, sys.getsizeof already includes the data buffer.
            # Only add nbytes for non-owning views where payload memory is external.
            try:
                if not obj.flags['OWNDATA']:
                    size += int(obj.nbytes)
            except Exception:
                pass
        elif hasattr(obj, 'nbytes'):
            try:
                size += int(obj.nbytes)
            except Exception:
                pass

        if _depth >= max_depth:
            return size

        if isinstance(obj, dict):
            for key, value in obj.items():
                size += self._estimate_object_bytes(key, max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
                size += self._estimate_object_bytes(value, max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                size += self._estimate_object_bytes(value, max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
        elif isinstance(obj, (set, frozenset)):
            for value in obj:
                size += self._estimate_object_bytes(value, max_depth=max_depth, _depth=_depth + 1, _seen=_seen)

        return size

    def run(self):
        """
        Run the training loop with mini-epoch validation and checkpointing.
        """
        self.logger.info(f"Starting training: epochs {self.start_epoch} to {self.num_epochs-1} (total {self.num_epochs} epochs)")
        self.logger.info(f"Mini-epoch: {self.mini_epoch_samples:,} samples ({self.mini_epoch_batches} batches of size {self.train_loader.batch_size})")
        
        batch_count = 0  # Initialize batch_count outside the loop
        
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{self.num_epochs}")
            
            # Check for shutdown before resetting datasets (which can be expensive)
            if self.shutdown_handler and self.shutdown_handler.shutdown_requested:
                self.logger.info("Shutdown requested before dataset reset, stopping training")
                raise GracefulShutdownRequested()
            
            # Reset datasets for new epoch (if they have a reset method)
            if hasattr(self.train_loader.dataset, 'reset'):
                self.logger.info("Resetting training dataset for new epoch")
                self.train_loader.dataset.reset()
            
            # Also reset validation dataset if it exists
            if self.val_loader and hasattr(self.val_loader.dataset, 'reset'):
                self.logger.info("Resetting validation dataset for new epoch")
                self.val_loader.dataset.reset()
            
            batch_iter = iter(self.train_loader)
            mini_epoch_idx = 0
            epoch_exhausted = False
            while True:
                try:
                    first_batch = next(batch_iter)
                    batch_count += 1
                except StopIteration:
                    self.logger.info(f"End of epoch {epoch+1} reached (StopIteration)")
                    self.logger.info(f"No more data in epoch {epoch+1}, breaking")
                    break  # No more data

                remaining_batches = self.mini_epoch_batches - 1

                # Stream mini-epoch batches instead of buffering the whole chunk.
                # This lowers peak memory during long-running training.
                def _mini_epoch_batch_stream():
                    nonlocal batch_count, epoch_exhausted
                    yield first_batch
                    for _ in range(remaining_batches):
                        try:
                            batch = next(batch_iter)
                            batch_count += 1
                            yield batch
                        except StopIteration:
                            self.logger.info(f"End of epoch {epoch+1} reached (StopIteration)")
                            epoch_exhausted = True
                            return
                
                # Validation (do this before training so we can pass metrics)
                val_metrics = None
                if self.val_loader is not None:
                    val_metrics = self.trainer.validate(epoch=epoch+1, mini_epoch=mini_epoch_idx+1)
                
                # Check for shutdown before starting training
                if self.shutdown_handler and self.shutdown_handler.shutdown_requested:
                    self.logger.info("Shutdown requested before training mini-epoch, stopping training")
                    raise GracefulShutdownRequested()
                
                # Train on this mini-epoch
                train_metrics = self.trainer.train_on_batches(
                    _mini_epoch_batch_stream(),
                    epoch=epoch+1,
                    mini_epoch=mini_epoch_idx+1,
                    val_metrics=val_metrics
                )
                
                # Checkpointing
                if self.checkpoint_dir is not None:
                    from hex_ai.file_utils import get_unique_checkpoint_path
                    from pathlib import Path
                    checkpoint_dir = Path(self.checkpoint_dir)
                    base_checkpoint_path = checkpoint_dir / f"epoch{epoch+1}_mini{mini_epoch_idx+1}.pt"
                    checkpoint_path = get_unique_checkpoint_path(base_checkpoint_path)
                    self.trainer.save_checkpoint(checkpoint_path, train_metrics, val_metrics, compress=True)
                
                # Logging
                if (mini_epoch_idx % self.log_interval == 0) or (mini_epoch_idx == 0):
                    msg = (
                        f"[Epoch {epoch+1}][Mini-epoch {mini_epoch_idx+1}] "
                        f"Train Losses: total={train_metrics.get('total_loss', float('nan')):.4f}, "
                        f"policy={train_metrics.get('policy_loss', float('nan')):.4f}, "
                        f"value={train_metrics.get('value_loss', float('nan')):.4f} "
                    )
                    if val_metrics:
                        msg += (
                            f"| Val Losses: total={val_metrics.get('total_loss', float('nan')):.4f}, "
                            f"policy={val_metrics.get('policy_loss', float('nan')):.4f}, "
                            f"value={val_metrics.get('value_loss', float('nan')):.4f} "
                        )
                    msg += f"| Batches processed: {batch_count}"
                    self.logger.info(msg)

                self._log_mini_epoch_memory_markers(epoch + 1, mini_epoch_idx + 1, batch_count)
                mini_epoch_idx += 1

                if epoch_exhausted:
                    self.logger.info(f"No more data in epoch {epoch+1}, breaking")
                    break
            
            # Take memory snapshot and write epoch summary after each epoch (if profiling enabled)
            profiler = get_profiler()
            if profiler is not None:
                profiler.take_snapshot(f"epoch_{epoch+1}_end")
                write_epoch_summary(epoch+1)
            
            # Analyze shared array memory at end of epoch (if diagnostics enabled)
            diagnostics = get_diagnostics()
            if diagnostics is not None and hasattr(self.train_loader.dataset, 'position_pool'):
                position_pool = self.train_loader.dataset.position_pool
                result = diagnostics.estimate_shared_array_memory(position_pool)
                
                if result['shared_position_count'] > 0:
                    diagnostics._log_diagnostic(
                        "📊 EPOCH MEMORY ANALYSIS",
                        f"Epoch {epoch+1} shared array analysis:\n"
                        f"  Positions with shared arrays: {result['shared_position_count']:,}\n"
                        f"  Estimated memory from shared arrays: {result['estimated_memory_mb']:.2f} MB ({result['estimated_memory_gb']:.2f} GB)\n"
                        f"  Total pool size: {len(position_pool):,} positions\n"
                        f"{'🚨 CONFIRMED: Shared arrays account for >2GB memory leak!' if result['estimated_memory_gb'] > 2.0 else '⚠️  Shared arrays <2GB, other sources may be responsible'}"
                    )
        
        self.logger.info(f"Training completed: processed {batch_count} total batches across {self.num_epochs - self.start_epoch} epochs")
        return {'total_batches': batch_count, 'epochs_completed': self.num_epochs - self.start_epoch} 
