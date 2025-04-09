"""
Base classes for UTCHS framework components.

This module provides base classes with proper initialization patterns
and standardized methods for UTCHS components to ensure consistency
and prevent initialization order issues.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
import json
import logging
from pathlib import Path

class UTCHSBaseComponent:
    """
    Base class for all UTCHS components.
    
    Provides common functionality and ensures proper initialization order.
    
    Attributes:
        logger: Component-specific logger
        name: Component name
        config: Configuration dictionary
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base component.
        
        Args:
            name: Component name
            config: Configuration dictionary
        """
        # Set up basic attributes first
        self.name = name
        self.config = config or {}
        
        # Set up logger
        self.logger = self._setup_logger()
        
        # Perform initialization in the right order
        self._pre_initialize()
        self._initialize()
        self._post_initialize()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up component-specific logger."""
        logger_name = f"utchs.{self.name}"
        return logging.getLogger(logger_name)
        
    def _pre_initialize(self) -> None:
        """Pre-initialization steps (override in subclasses)."""
        pass
        
    def _initialize(self) -> None:
        """Main initialization steps (override in subclasses)."""
        pass
        
    def _post_initialize(self) -> None:
        """Post-initialization steps (override in subclasses)."""
        pass
        
    def validate_config(self, required_params: List[str]) -> None:
        """
        Validate component configuration.
        
        Args:
            required_params: List of required parameter names
            
        Raises:
            ValueError: If a required parameter is missing
        """
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
                
        self.logger.debug("Configuration validation successful")
        
    def get_serializable_state(self) -> Dict[str, Any]:
        """
        Get a serializable representation of component state.
        
        Returns:
            Dictionary with serializable state
        """
        return {
            "name": self.name,
            "config": self._get_serializable_config()
        }
    
    def _get_serializable_config(self) -> Dict[str, Any]:
        """
        Convert configuration to a serializable format.
        
        Returns:
            Serializable configuration dictionary
        """
        def make_serializable(obj):
            """Make an object JSON serializable."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, complex):
                return {"real": obj.real, "imag": obj.imag}
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return obj
            
        return make_serializable(self.config)
    
    def _safe_numpy_operation(self, operation, *args, **kwargs):
        """
        Safely perform a numpy operation with error handling.
        
        Args:
            operation: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation
            
        Raises:
            RuntimeError: If the operation fails
        """
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Numpy operation failed: {str(e)}")
            raise RuntimeError(f"Error in numpy operation: {str(e)}") from e


class FieldBase(UTCHSBaseComponent):
    """
    Base class for field components.
    
    Provides common functionality for field operations.
    
    Attributes:
        grid_size: Size of the field grid
        dx: Grid spacing
        field: Field data
        field_history: History of field states
    """
    
    def _pre_initialize(self) -> None:
        """Pre-initialization for fields."""
        # Set field attributes before any field operations
        self.grid_size = tuple(self.config.get('grid_size', (50, 50, 50)))
        self.dx = self.config.get('grid_spacing', 0.1)
        
        # Initialize field history BEFORE any other field operations
        self.history_length = self.config.get('history_length', 10)
        self.field_history = []
        
        # Initialize field to zeros by default
        self.field = None  # Will be set in _initialize()
    
    def update_history(self) -> None:
        """
        Update field history.
        
        This method is now safe to call at any point after initialization.
        """
        if hasattr(self, 'field_history') and self.field is not None:
            self.field_history.append(self.field.copy())
            
            # Limit history length
            while len(self.field_history) > self.history_length:
                self.field_history.pop(0)
    
    def get_history_snapshot(self, index: int = -1) -> Optional[np.ndarray]:
        """
        Get a snapshot from the field history.
        
        Args:
            index: History index (-1 for most recent)
            
        Returns:
            Field snapshot or None if index is invalid
        """
        if not hasattr(self, 'field_history') or not self.field_history:
            return None
            
        if abs(index) > len(self.field_history):
            return None
            
        return self.field_history[index] 