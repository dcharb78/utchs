"""Unit tests for the phase field module."""

import numpy as np
import pytest
from utchs.fields.phase_field import PhaseField
from utchs.math.mobius import MobiusTransformation


@pytest.fixture
def phase_field():
    """Create a phase field fixture."""
    return PhaseField((10, 10, 10))


def test_phase_field_initialization():
    """Test phase field initialization."""
    field = PhaseField((10, 10, 10))
    assert field.field.shape == (10, 10, 10)
    assert field.grid_size == (10, 10, 10)
    assert isinstance(field.field, np.ndarray)
    assert field.field.dtype == np.complex128


def test_phase_circulation(phase_field):
    """Test phase circulation calculation."""
    # Create a simple vortex field
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z = np.linspace(-1, 1, 10)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    theta = np.arctan2(Y, X)
    phase_field.field = np.exp(1j * theta)
    
    # Test circulation at center
    circulation = phase_field._calculate_phase_circulation((5, 5, 5))
    assert abs(abs(circulation) - 2*np.pi) < 0.1


def test_singularity_detection(phase_field):
    """Test singularity detection."""
    # Create field with known singularity
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z = np.linspace(-1, 1, 10)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    theta = np.arctan2(Y, X)
    phase_field.field = np.exp(1j * theta)
    
    # Find singularities
    phase_field._find_singularities()
    assert len(phase_field.singularities) > 0
    assert all('position' in s for s in phase_field.singularities)
    assert all('charge' in s for s in phase_field.singularities)


def test_field_update(phase_field):
    """Test field update with Möbius transformation."""
    # Create identity transformation
    transform = MobiusTransformation(1, 0, 0, 1)
    
    # Store original field
    original_field = phase_field.field.copy()
    
    # Update field
    phase_field.update(transform)
    
    # Field should remain unchanged
    np.testing.assert_array_almost_equal(phase_field.field, original_field) 