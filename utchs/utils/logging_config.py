"""
Logging configuration module for the UTCHS framework.

This module provides centralized logging configuration and utility functions
for consistent logging across the framework.
"""

import os
import logging
import logging.handlers
import sys
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

from utchs.core.exceptions import ConfigurationError

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Log levels
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

class LoggingConfig:
    """Configuration for logging in the UTCHS framework."""

    def __init__(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_format: Optional[str] = None,
        console_output: bool = True,
        file_output: bool = True,
        log_dir: str = "logs",
    ):
        """Initialize the logging configuration.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file
            log_format: Custom log format
            console_output: Whether to output logs to console
            file_output: Whether to output logs to file
            log_dir: Directory for log files
        """
        self.log_level = log_level.upper()
        self.log_file = log_file
        self.log_format = log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.console_output = console_output
        self.file_output = file_output
        self.log_dir = log_dir
        
        # Validate log level
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigurationError(f"Invalid log level: {self.log_level}")
        
        # Create log directory if needed
        if self.file_output and not self.log_file:
            os.makedirs(self.log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(self.log_dir, f"utchs_{timestamp}.log")
        
        # Configure logging
        self._configure_logging()
    
    def _configure_logging(self) -> None:
        """Configure the logging system."""
        # Get the root logger
        logger = logging.getLogger()
        logger.setLevel(self.log_level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(self.log_format)
        
        # Add console handler if needed
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add file handler if needed
        if self.file_output and self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name.

        Args:
            name: Logger name

        Returns:
            Configured logger
        """
        return logging.getLogger(name)
    
    def set_log_level(self, level: str) -> None:
        """Set the logging level.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Raises:
            ConfigurationError: If level is invalid
        """
        level = level.upper()
        if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigurationError(f"Invalid log level: {level}")
        
        self.log_level = level
        logging.getLogger().setLevel(level)
    
    def add_file_handler(self, log_file: str) -> None:
        """Add a file handler to the logging configuration.

        Args:
            log_file: Path to log file
        """
        logger = logging.getLogger()
        formatter = logging.Formatter(self.log_format)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def remove_file_handler(self, log_file: str) -> None:
        """Remove a file handler from the logging configuration.

        Args:
            log_file: Path to log file
        """
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file:
                logger.removeHandler(handler)


# Create a default logging configuration
default_logging_config = LoggingConfig()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Configured logger
    """
    return default_logging_config.get_logger(name)


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    log_dir: str = "logs",
) -> LoggingConfig:
    """Configure the logging system.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Custom log format
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        log_dir: Directory for log files

    Returns:
        Logging configuration
    """
    global default_logging_config
    default_logging_config = LoggingConfig(
        log_level=log_level,
        log_file=log_file,
        log_format=log_format,
        console_output=console_output,
        file_output=file_output,
        log_dir=log_dir,
    )
    return default_logging_config

class UTCHSError(Exception):
    """Base exception class for UTCHS framework."""
    pass

class ConfigurationError(UTCHSError):
    """Exception raised for configuration-related errors."""
    pass

class FieldError(UTCHSError):
    """Exception raised for field-related errors."""
    pass

class SystemError(UTCHSError):
    """Exception raised for system-related errors."""
    pass

class ValidationError(UTCHSError):
    """Exception raised for validation errors."""
    pass 