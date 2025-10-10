"""
Model cache manager for tournament play.

This module provides efficient model caching to avoid reloading models
for every move during tournaments.
"""

import os
from typing import Dict, Optional, Tuple
from pathlib import Path

from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.inference.model_config import get_normalized_path


class TemporaryModelCache:
    """Temporary model cache for per-match model loading."""
    
    def __init__(self, models_dict: Dict[str, SimpleModelInference]):
        self.models = models_dict
    
    def get_simple_model(self, checkpoint_path: str, verbose: int = 1) -> SimpleModelInference:
        """Get a model from the temporary cache."""
        normalized_path = get_normalized_path(checkpoint_path)
        return self.models[normalized_path]


class ModelCache:
    """
    Cache for model instances to avoid reloading during tournaments.
    
    This class manages both SimpleModelInference and ModelWrapper instances,
    providing the appropriate type based on the strategy requirements.
    """
    
    def __init__(self):
        self._simple_models: Dict[str, SimpleModelInference] = {}
        self._wrapper_models: Dict[str, ModelWrapper] = {}
    
    def get_simple_model(self, checkpoint_path: str, verbose: int = 1) -> SimpleModelInference:
        """Get or create a SimpleModelInference instance."""
        normalized_path = get_normalized_path(checkpoint_path)
        if normalized_path not in self._simple_models:
            self._simple_models[normalized_path] = SimpleModelInference(checkpoint_path, verbose=verbose)
        return self._simple_models[normalized_path]
    
    def get_wrapper_model(self, checkpoint_path: str) -> ModelWrapper:
        """Get or create a ModelWrapper instance."""
        normalized_path = get_normalized_path(checkpoint_path)
        if normalized_path not in self._wrapper_models:
            # Get the simple model first to extract model_type
            simple_model = self.get_simple_model(checkpoint_path)
            self._wrapper_models[normalized_path] = ModelWrapper(
                checkpoint_path, 
                device=None, 
                model_type=simple_model.model_type
            )
        return self._wrapper_models[normalized_path]
    
    def clear_cache(self) -> None:
        """Clear all cached models to free memory."""
        self._simple_models.clear()
        self._wrapper_models.clear()
    
    def get_temporary_models(self, checkpoint_paths: list, verbose: int = 1) -> Dict[str, SimpleModelInference]:
        """
        Get models temporarily for a match, without caching them.
        
        Args:
            checkpoint_paths: List of model paths to load
            verbose: Verbosity level for model loading
            
        Returns:
            Dictionary mapping normalized paths to SimpleModelInference instances
        """
        models = {}
        for path in checkpoint_paths:
            normalized_path = get_normalized_path(path)
            models[normalized_path] = SimpleModelInference(path, verbose=verbose)
        return models


# Global cache instance
_model_cache = ModelCache()


def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    return _model_cache


def clear_tournament_cache() -> None:
    """Clear the tournament model cache."""
    _model_cache.clear_cache()

def get_temporary_models_for_match(checkpoint_paths: list, verbose: int = 1) -> Dict[str, SimpleModelInference]:
    """Get models temporarily for a match, without caching them."""
    return _model_cache.get_temporary_models(checkpoint_paths, verbose)

def create_temporary_model_cache(checkpoint_paths: list, verbose: int = 1) -> TemporaryModelCache:
    """Create a temporary model cache for a match."""
    temporary_models = get_temporary_models_for_match(checkpoint_paths, verbose)
    return TemporaryModelCache(temporary_models)
