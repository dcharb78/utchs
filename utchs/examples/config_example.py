"""Example demonstrating the configuration management system."""

import os
from pathlib import Path

from utchs.config.config_manager import ConfigManager
from utchs.core.exceptions import ConfigurationError


def main():
    """Run the configuration example."""
    # Create a new configuration manager with default settings
    config_manager = ConfigManager()
    
    # Get the current configuration
    config = config_manager.get_config()
    print("Default configuration:")
    print(f"Dimensions: {config.dimensions}")
    print(f"Lattice size: {config.lattice_size}")
    print(f"Field type: {config.field_type}")
    print()
    
    # Update some configuration values
    updates = {
        "dimensions": 2,
        "lattice_size": 50,
        "field_type": "energy",
        "field_parameters": {
            "amplitude": 2.0,
            "phase": 0.5,
            "wavelength": 2.0
        }
    }
    
    try:
        config_manager.update_config(updates)
        print("Updated configuration:")
        print(f"Dimensions: {config_manager.get_value('dimensions')}")
        print(f"Lattice size: {config_manager.get_value('lattice_size')}")
        print(f"Field type: {config_manager.get_value('field_type')}")
        print(f"Field parameters: {config_manager.get_value('field_parameters')}")
        print()
    except ConfigurationError as e:
        print(f"Error updating configuration: {e}")
    
    # Save the configuration to a file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    config_path = output_dir / "example_config.yaml"
    
    try:
        config_manager.save_config(str(config_path))
        print(f"Configuration saved to {config_path}")
        print()
    except ConfigurationError as e:
        print(f"Error saving configuration: {e}")
    
    # Load the configuration from the file
    try:
        new_config_manager = ConfigManager(str(config_path))
        new_config = new_config_manager.get_config()
        print("Loaded configuration:")
        print(f"Dimensions: {new_config.dimensions}")
        print(f"Lattice size: {new_config.lattice_size}")
        print(f"Field type: {new_config.field_type}")
        print(f"Field parameters: {new_config.field_parameters}")
        print()
    except ConfigurationError as e:
        print(f"Error loading configuration: {e}")
    
    # Demonstrate validation
    try:
        # This should fail
        config_manager.set_value("dimensions", 4)
        config_manager.validate_config()
    except ConfigurationError as e:
        print(f"Validation error (expected): {e}")
        print()
    
    # Reset to valid configuration
    config_manager.set_value("dimensions", 2)
    try:
        config_manager.validate_config()
        print("Configuration validation successful")
    except ConfigurationError as e:
        print(f"Validation error: {e}")


if __name__ == "__main__":
    main() 