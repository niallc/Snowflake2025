import logging
from typing import Optional
from hex_ai.error_handling import GracefulShutdownRequested

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

    def run(self):
        """
        Run the training loop with mini-epoch validation and checkpointing.
        """
        self.logger.info(f"Starting training: epochs {self.start_epoch} to {self.num_epochs-1} (total {self.num_epochs} epochs)")
        self.logger.info(f"Mini-epoch: {self.mini_epoch_samples:,} samples ({self.mini_epoch_batches} batches of size {self.train_loader.batch_size})")
        
        batch_count = 0  # Initialize batch_count outside the loop
        
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{self.num_epochs}")
            batch_iter = iter(self.train_loader)
            mini_epoch_idx = 0
            while True:
                mini_epoch_batches = []
                try:
                    for _ in range(self.mini_epoch_batches):
                        mini_epoch_batches.append(next(batch_iter))
                        batch_count += 1
                except StopIteration:
                    self.logger.info(f"End of epoch {epoch+1} reached (StopIteration)")
                    pass  # End of epoch
                if not mini_epoch_batches:
                    self.logger.info(f"No more data in epoch {epoch+1}, breaking")
                    break  # No more data
                
                # Validation (do this before training so we can pass metrics)
                val_metrics = None
                if self.val_loader is not None:
                    val_metrics = self.trainer.validate()
                
                # Train on this mini-epoch
                train_metrics = self.trainer.train_on_batches(mini_epoch_batches, epoch=epoch, mini_epoch=mini_epoch_idx, val_metrics=val_metrics)
                
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
                    print(msg)
                    self.logger.info(msg)
                mini_epoch_idx += 1
                # Graceful shutdown check
                if self.shutdown_handler and self.shutdown_handler.shutdown_requested:
                    self.logger.info("Shutdown requested, stopping training")
                    raise GracefulShutdownRequested()
        
        self.logger.info(f"Training completed: processed {batch_count} total batches across {self.num_epochs - self.start_epoch} epochs")
        return {'total_batches': batch_count, 'epochs_completed': self.num_epochs - self.start_epoch} 