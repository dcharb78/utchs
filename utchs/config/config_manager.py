"""Configuration management system for UTCHS."""

import os
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError

from utchs.core.exceptions import ConfigurationError


class SystemConfig(BaseModel):
    """System configuration model."""

    # System parameters
    dimensions: int = Field(default=3, ge=1, le=3)
    periodic_boundary: bool = Field(default=True)
    lattice_size: int = Field(default=100, gt=0)
    time_steps: int = Field(default=1000, gt=0)
    temperature: float = Field(default=0.1, gt=0)
    
    # Field parameters
    field_type: str = Field(default="phase")
    field_parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Output parameters
    output_dir: str = Field(default="output")
    save_interval: int = Field(default=100, gt=0)
    log_level: str = Field(default="INFO")
    
    # Performance parameters
    num_threads: int = Field(default=1, gt=0)
    use_gpu: bool = Field(default=False)
    batch_size: int = Field(default=32, gt=0)


class ConfigManager:
    """Configuration manager for UTCHS."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Optional[SystemConfig] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file or use defaults."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                self.config = SystemConfig(**config_data)
            except (yaml.YAMLError, ValidationError) as e:
                raise ConfigurationError(f"Error loading configuration: {str(e)}")
        else:
            self.config = SystemConfig()

    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file.

        Args:
            path: Path to save configuration file
        """
        if not self.config:
            raise ConfigurationError("No configuration to save")
        
        save_path = path or self.config_path
        if not save_path:
            raise ConfigurationError("No path specified for saving configuration")
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config.dict(), f)
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {str(e)}")

    def get_config(self) -> SystemConfig:
        """Get current configuration.

        Returns:
            Current system configuration

        Raises:
            ConfigurationError: If configuration is not loaded
        """
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        return self.config

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates

        Raises:
            ConfigurationError: If update fails
        """
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        
        try:
            current_config = self.config.dict()
            current_config.update(updates)
            self.config = SystemConfig(**current_config)
        except ValidationError as e:
            raise ConfigurationError(f"Invalid configuration update: {str(e)}")

    def validate_config(self) -> None:
        """Validate current configuration.

        Raises:
            ConfigurationError: If validation fails
        """
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        
        try:
            # Pydantic validation is automatic, but we can add custom validation here
            if self.config.dimensions not in [1, 2, 3]:
                raise ConfigurationError("Dimensions must be 1, 2, or 3")
            
            if self.config.field_type not in ["phase", "energy"]:
                raise ConfigurationError("Field type must be 'phase' or 'energy'")
            
            if self.config.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                raise ConfigurationError("Invalid log level")
            
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")

    def get_value(self, key: str) -> Any:
        """Get a specific configuration value.

        Args:
            key: Configuration key

        Returns:
            Configuration value

        Raises:
            ConfigurationError: If key doesn't exist
        """
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        
        try:
            return getattr(self.config, key)
        except AttributeError:
            raise ConfigurationError(f"Configuration key '{key}' not found")

    def set_value(self, key: str, value: Any) -> None:
        """Set a specific configuration value.

        Args:
            key: Configuration key
            value: New value

        Raises:
            ConfigurationError: If update fails
        """
        self.update_config({key: value}) 