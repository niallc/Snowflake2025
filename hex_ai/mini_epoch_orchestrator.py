import logging
from typing import Optional
from hex_ai.error_handling import GracefulShutdownRequested

class MiniEpochOrchestrator:
    """
    Orchestrates training in mini-epochs, enabling validation and checkpointing at configurable batch intervals.

    This class wraps a Trainer and DataLoader, running training in mini-epochs (N batches), and performing
    validation/checkpointing after each mini-epoch. It maintains model/optimizer state across the entire run.

    Args:
        trainer: The Trainer instance (must provide train_on_batches, validate, save_checkpoint).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data (optional).
        mini_epoch_batches: Number of batches per mini-epoch (int).
        num_epochs: Number of full epochs to train.
        checkpoint_dir: Directory to save checkpoints (str or Path).
        log_interval: How often to log progress (in mini-epochs).

    Usage:
        orchestrator = MiniEpochOrchestrator(trainer, train_loader, val_loader, mini_epoch_batches=500, num_epochs=10)
        orchestrator.run()
    """
    def __init__(self, trainer, train_loader, val_loader=None, mini_epoch_batches=500, num_epochs=1, checkpoint_dir=None, log_interval=1, shutdown_handler=None):
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mini_epoch_batches = mini_epoch_batches
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        self.shutdown_handler = shutdown_handler

    def run(self):
        """
        Run the training loop with mini-epoch validation and checkpointing.
        """
        for epoch in range(self.num_epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{self.num_epochs}")
            batch_iter = iter(self.train_loader)
            batch_count = 0
            mini_epoch_idx = 0
            while True:
                mini_epoch_batches = []
                try:
                    for _ in range(self.mini_epoch_batches):
                        mini_epoch_batches.append(next(batch_iter))
                        batch_count += 1
                except StopIteration:
                    pass  # End of epoch
                if not mini_epoch_batches:
                    break  # No more data
                # Train on this mini-epoch
                train_metrics = self.trainer.train_on_batches(mini_epoch_batches, epoch=epoch, mini_epoch=mini_epoch_idx)
                # Validation
                val_metrics = None
                if self.val_loader is not None:
                    val_metrics = self.trainer.validate()
                # Checkpointing
                if self.checkpoint_dir is not None:
                    checkpoint_path = f"{self.checkpoint_dir}/epoch{epoch+1}_mini{mini_epoch_idx+1}.pt"
                    self.trainer.save_checkpoint(checkpoint_path, train_metrics, val_metrics)
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
                    # CSV logging
                    if hasattr(self.trainer, 'csv_logger') and self.trainer.csv_logger is not None:
                        # Extract hyperparameters as in Trainer
                        hp = {
                            'learning_rate': self.trainer.optimizer.param_groups[0]['lr'],
                            'batch_size': self.trainer.train_loader.batch_size,
                            'dataset_size': 'N/A',
                            'network_structure': f"ResNet{getattr(self.trainer.model, 'resnet_depth', '?')}",
                            'policy_weight': getattr(self.trainer.criterion, 'policy_weight', ''),
                            'value_weight': getattr(self.trainer.criterion, 'value_weight', ''),
                            'total_loss_weight': getattr(self.trainer.criterion, 'policy_weight', 0) + getattr(self.trainer.criterion, 'value_weight', 0),
                            'dropout_prob': getattr(self.trainer.model, 'dropout', type('dummy', (), {'p': ''})) .p if hasattr(self.trainer.model, 'dropout') else '',
                            'weight_decay': self.trainer.optimizer.param_groups[0].get('weight_decay', 0.0)
                        }
                        epoch_id = f"{epoch+1}_mini{mini_epoch_idx+1}"
                        self.trainer.csv_logger.log_mini_epoch(
                            epoch=epoch_id,
                            train_metrics=train_metrics,
                            val_metrics=val_metrics,
                            hyperparams=hp,
                            training_time=0.0,
                            epoch_time=0.0,
                            samples_per_second=0.0,
                            memory_usage_mb=0.0,
                            gpu_memory_mb=None,
                            gradient_norm=None,
                            weight_stats=None,
                            notes="mini-epoch"
                        )
                mini_epoch_idx += 1
                # Graceful shutdown check
                if self.shutdown_handler is not None and self.shutdown_handler.shutdown_requested:
                    self.logger.info("Graceful shutdown requested. Exiting after current mini-epoch.")
                    raise GracefulShutdownRequested()
                if len(mini_epoch_batches) < self.mini_epoch_batches:
                    break  # End of epoch 