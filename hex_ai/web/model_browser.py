"""
Web-specific model browser functionality.

This module provides utilities for the web interface to browse and select model files.
It builds on the generic utilities in hex_ai.file_utils.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from hex_ai.file_utils import (
    scan_checkpoint_directory,
    validate_model_file,
    get_model_directories,
    get_models_in_directory,
    load_recent_models,
    add_recent_model
)
from hex_ai.inference.model_config import get_all_model_info, get_model_path, is_valid_model_id

logger = logging.getLogger(__name__)


class ModelBrowser:
    """Web interface for browsing and selecting model files."""
    
    def __init__(self, checkpoints_dir: str = "checkpoints"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self._cached_models = None
        self._cache_timestamp = None
    
    def get_recent_models(self) -> List[Dict[str, Any]]:
        """
        Get list of recently used models with validation.
        
        Returns:
            List of model info dictionaries for recent models
        """
        recent_paths = load_recent_models()
        recent_models = []
        
        for model_path in recent_paths:
            # Validate the model still exists
            is_valid, error_msg = validate_model_file(model_path)
            if is_valid:
                # Get full model info
                model_info = self._get_model_info_by_path(model_path)
                if model_info:
                    recent_models.append(model_info)
            else:
                logger.debug(f"Recent model no longer valid: {model_path} - {error_msg}")
        
        return recent_models
    
    def get_all_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get all available models, with caching.
        
        This integrates with the centralized model configuration to ensure
        consistency with other parts of the system.
        
        Args:
            force_refresh: Force refresh of cached data
            
        Returns:
            List of all model info dictionaries
        """
        if not force_refresh and self._cached_models is not None:
            return self._cached_models
        
        try:
            # First get models from centralized configuration
            centralized_models = get_all_model_info()
            
            # Convert centralized models to relative paths for browser compatibility
            centralized_models_relative = []
            for model in centralized_models:
                # Convert absolute path to relative path for browser compatibility
                if model['path'].startswith('checkpoints/'):
                    relative_path = model['path'][len('checkpoints/'):]
                    model_copy = model.copy()
                    model_copy['path'] = relative_path
                    model_copy['relative_path'] = relative_path
                    centralized_models_relative.append(model_copy)
                else:
                    centralized_models_relative.append(model)
            
            # Then scan for additional models in checkpoints directory
            scanned_models = scan_checkpoint_directory(self.checkpoints_dir)
            
            # Combine and deduplicate based on relative paths
            all_models = centralized_models_relative.copy()
            centralized_relative_paths = {model['relative_path'] for model in centralized_models_relative}
            
            # Add scanned models that aren't already in centralized config
            for scanned_model in scanned_models:
                if scanned_model['relative_path'] not in centralized_relative_paths:
                    all_models.append(scanned_model)
            
            self._cached_models = all_models
            return all_models
        except Exception as e:
            logger.error(f"Error getting all models: {e}")
            return []
    
    def get_directories(self) -> List[str]:
        """
        Get list of all directories containing models.
        
        Returns:
            List of directory names
        """
        try:
            return get_model_directories(self.checkpoints_dir)
        except Exception as e:
            logger.error(f"Error getting directories: {e}")
            return []
    
    def get_models_in_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        Get all models in a specific directory.
        
        Args:
            directory: Directory name (relative to checkpoints/)
            
        Returns:
            List of model info dictionaries
        """
        try:
            return get_models_in_directory(directory, self.checkpoints_dir)
        except Exception as e:
            logger.error(f"Error getting models in directory {directory}: {e}")
            return []
    
    def validate_model(self, model_path: str) -> Dict[str, Any]:
        """
        Validate a model path and return validation result.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary with validation result
        """
        is_valid, error_msg = validate_model_file(model_path)
        
        result = {
            'valid': is_valid,
            'error': error_msg if not is_valid else None
        }
        
        if is_valid:
            # Add model info
            model_info = self._get_model_info_by_path(model_path)
            if model_info:
                result.update(model_info)
        
        return result
    
    def select_model(self, model_path: str) -> Dict[str, Any]:
        """
        Select a model and add it to recent models.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary with selection result
        """
        # Validate the model
        validation = self.validate_model(model_path)
        
        if not validation['valid']:
            return {
                'success': False,
                'error': validation['error']
            }
        
        try:
            # Add to recent models
            add_recent_model(model_path)
            
            return {
                'success': True,
                'model_info': validation
            }
            
        except Exception as e:
            logger.error(f"Error selecting model {model_path}: {e}")
            return {
                'success': False,
                'error': f"Error selecting model: {e}"
            }
    
    def search_models(self, query: str) -> List[Dict[str, Any]]:
        """
        Search models by directory name, filename, or path.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching model info dictionaries
        """
        if not query.strip():
            return []
        
        query = query.lower()
        all_models = self.get_all_models()
        matches = []
        
        for model in all_models:
            # Search in directory name, filename, and full path
            searchable_text = f"{model['directory']} {model['filename']} {model['relative_path']}".lower()
            
            if query in searchable_text:
                matches.append(model)
        
        return matches
    
    def _get_model_info_by_path(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        Get model info by path from cached data.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Model info dictionary or None if not found
        """
        all_models = self.get_all_models()
        
        # Try to find by relative path first
        for model in all_models:
            if model['relative_path'] == model_path:
                return model
        
        # Try to find by full path
        for model in all_models:
            if model['path'] == model_path:
                return model
        
        return None


def create_model_browser() -> ModelBrowser:
    """
    Create a ModelBrowser instance with default settings.
    
    Returns:
        ModelBrowser instance
    """
    return ModelBrowser() 