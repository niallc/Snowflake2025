"""
Model cache manager for tournament play.

This module provides efficient model caching to avoid reloading models
for every move during tournaments.
"""

from typing import Dict, Optional, Tuple
from pathlib import Path

from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.model_wrapper import ModelWrapper


class ModelCache:
    """
    Cache for model instances to avoid reloading during tournaments.
    
    This class manages both SimpleModelInference and ModelWrapper instances,
    providing the appropriate type based on the strategy requirements.
    """
    
    def __init__(self):
        self._simple_models: Dict[str, SimpleModelInference] = {}
        self._wrapper_models: Dict[str, ModelWrapper] = {}
    
    def get_simple_model(self, checkpoint_path: str) -> SimpleModelInference:
        """Get or create a SimpleModelInference instance."""
        if checkpoint_path not in self._simple_models:
            self._simple_models[checkpoint_path] = SimpleModelInference(checkpoint_path)
        return self._simple_models[checkpoint_path]
    
    def get_wrapper_model(self, checkpoint_path: str) -> ModelWrapper:
        """Get or create a ModelWrapper instance."""
        if checkpoint_path not in self._wrapper_models:
            # Get the simple model first to extract model_type
            simple_model = self.get_simple_model(checkpoint_path)
            self._wrapper_models[checkpoint_path] = ModelWrapper(
                checkpoint_path, 
                device=None, 
                model_type=simple_model.model_type
            )
        return self._wrapper_models[checkpoint_path]
    
    def preload_models(self, checkpoint_paths: list) -> None:
        """Preload all models for a tournament."""
        print(f"Preloading {len(checkpoint_paths)} models...")
        for path in checkpoint_paths:
            # Load both types to ensure they're cached
            self.get_simple_model(path)
            self.get_wrapper_model(path)
        print("Model preloading complete.")
    
    def clear_cache(self) -> None:
        """Clear all cached models to free memory."""
        self._simple_models.clear()
        self._wrapper_models.clear()


# Global cache instance
_model_cache = ModelCache()


def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    return _model_cache


def preload_tournament_models(checkpoint_paths: list) -> None:
    """Preload models for a tournament."""
    _model_cache.preload_models(checkpoint_paths)


def clear_tournament_cache() -> None:
    """Clear the tournament model cache."""
    _model_cache.clear_cache()
