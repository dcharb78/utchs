"""Unit tests for the configuration manager."""

import os
import pytest
import tempfile
from pathlib import Path

from utchs.config.config_manager import ConfigManager, SystemConfig
from utchs.core.exceptions import ConfigurationError


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
        dimensions: 2
        periodic_boundary: true
        lattice_size: 50
        time_steps: 500
        temperature: 0.2
        field_type: "phase"
        field_parameters:
          amplitude: 2.0
          phase: 0.5
        output_dir: "test_output"
        save_interval: 50
        log_level: "DEBUG"
        num_threads: 2
        use_gpu: true
        batch_size: 64
        """)
    yield f.name
    os.unlink(f.name)


def test_default_config():
    """Test default configuration."""
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    assert isinstance(config, SystemConfig)
    assert config.dimensions == 3
    assert config.periodic_boundary is True
    assert config.lattice_size == 100
    assert config.time_steps == 1000
    assert config.temperature == 0.1
    assert config.field_type == "phase"
    assert config.output_dir == "output"
    assert config.log_level == "INFO"
    assert config.num_threads == 1
    assert config.use_gpu is False
    assert config.batch_size == 32


def test_load_config(temp_config_file):
    """Test loading configuration from file."""
    config_manager = ConfigManager(temp_config_file)
    config = config_manager.get_config()
    
    assert config.dimensions == 2
    assert config.periodic_boundary is True
    assert config.lattice_size == 50
    assert config.time_steps == 500
    assert config.temperature == 0.2
    assert config.field_type == "phase"
    assert config.field_parameters["amplitude"] == 2.0
    assert config.field_parameters["phase"] == 0.5
    assert config.output_dir == "test_output"
    assert config.save_interval == 50
    assert config.log_level == "DEBUG"
    assert config.num_threads == 2
    assert config.use_gpu is True
    assert config.batch_size == 64


def test_save_config(temp_config_file):
    """Test saving configuration to file."""
    config_manager = ConfigManager(temp_config_file)
    
    # Modify config
    config_manager.set_value("dimensions", 1)
    config_manager.set_value("lattice_size", 25)
    
    # Save to new file
    new_config_path = temp_config_file + ".new"
    config_manager.save_config(new_config_path)
    
    # Load from new file
    new_config_manager = ConfigManager(new_config_path)
    new_config = new_config_manager.get_config()
    
    assert new_config.dimensions == 1
    assert new_config.lattice_size == 25
    
    os.unlink(new_config_path)


def test_update_config():
    """Test updating configuration."""
    config_manager = ConfigManager()
    
    updates = {
        "dimensions": 2,
        "lattice_size": 50,
        "field_type": "energy"
    }
    
    config_manager.update_config(updates)
    config = config_manager.get_config()
    
    assert config.dimensions == 2
    assert config.lattice_size == 50
    assert config.field_type == "energy"


def test_validate_config():
    """Test configuration validation."""
    config_manager = ConfigManager()
    
    # Valid config
    config_manager.validate_config()
    
    # Invalid dimensions
    with pytest.raises(ConfigurationError):
        config_manager.set_value("dimensions", 4)
        config_manager.validate_config()
    
    # Invalid field type
    with pytest.raises(ConfigurationError):
        config_manager.set_value("field_type", "invalid")
        config_manager.validate_config()
    
    # Invalid log level
    with pytest.raises(ConfigurationError):
        config_manager.set_value("log_level", "INVALID")
        config_manager.validate_config()


def test_get_set_value():
    """Test getting and setting configuration values."""
    config_manager = ConfigManager()
    
    # Get value
    assert config_manager.get_value("dimensions") == 3
    
    # Set value
    config_manager.set_value("dimensions", 2)
    assert config_manager.get_value("dimensions") == 2
    
    # Get non-existent value
    with pytest.raises(ConfigurationError):
        config_manager.get_value("non_existent")


def test_invalid_config_file():
    """Test handling of invalid configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
    
    with pytest.raises(ConfigurationError):
        ConfigManager(f.name)
    
    os.unlink(f.name)


def test_missing_config_file():
    """Test handling of missing configuration file."""
    config_manager = ConfigManager("non_existent.yaml")
    config = config_manager.get_config()
    
    # Should use default config
    assert config.dimensions == 3
    assert config.lattice_size == 100 