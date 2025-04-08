"""Integration tests for field system interactions."""

import numpy as np
import pytest
from utchs.fields.phase_field import PhaseField
from utchs.math.mobius import MobiusTransformation
from utchs.core.system import UTCHSSystem
from utchs.config import load_config


@pytest.fixture
def system():
    """Create a UTCHS system fixture."""
    config = load_config()
    return UTCHSSystem(config)


def test_field_initialization_workflow(system):
    """Test complete field initialization workflow."""
    # Check phase field initialization
    assert isinstance(system.phase_field, PhaseField)
    assert system.phase_field.field.shape == system.config['grid_size']
    
    # Check energy field initialization
    assert isinstance(system.energy_field, np.ndarray)
    assert system.energy_field.shape == system.config['grid_size']


def test_field_update_workflow(system):
    """Test complete field update workflow."""
    # Create a Möbius transformation
    transform = MobiusTransformation(1, 0, 0, 1)
    
    # Store initial state
    initial_phase = system.phase_field.field.copy()
    initial_energy = system.energy_field.copy()
    
    # Update system
    system.update(transform)
    
    # Check phase field update
    assert not np.array_equal(system.phase_field.field, initial_phase)
    
    # Check energy field update
    assert not np.array_equal(system.energy_field, initial_energy)


def test_singularity_detection_workflow(system):
    """Test complete singularity detection workflow."""
    # Create a field with known singularities
    x = np.linspace(-1, 1, system.config['grid_size'][0])
    y = np.linspace(-1, 1, system.config['grid_size'][1])
    z = np.linspace(-1, 1, system.config['grid_size'][2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    theta = np.arctan2(Y, X)
    system.phase_field.field = np.exp(1j * theta)
    
    # Find singularities
    system.phase_field._find_singularities()
    
    # Check singularity detection
    assert len(system.phase_field.singularities) > 0
    assert all('position' in s for s in system.phase_field.singularities)
    assert all('charge' in s for s in system.phase_field.singularities)


def test_system_stability(system):
    """Test system stability over multiple updates."""
    # Create a sequence of transformations
    transforms = [
        MobiusTransformation(1, 0, 0, 1),
        MobiusTransformation(0.5, 0, 0, 2),
        MobiusTransformation(2, 0, 0, 0.5)
    ]
    
    # Apply transformations
    for transform in transforms:
        system.update(transform)
        
        # Check field validity
        assert np.all(np.isfinite(system.phase_field.field))
        assert np.all(np.isfinite(system.energy_field))
        
        # Check energy conservation
        assert np.all(system.energy_field >= 0) 