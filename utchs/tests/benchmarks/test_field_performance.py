"""Benchmark tests for field operations."""

import numpy as np
import pytest
from utchs.fields.phase_field import PhaseField
from utchs.math.mobius import MobiusTransformation


@pytest.fixture
def large_field():
    """Create a large phase field fixture."""
    return PhaseField((100, 100, 100))


def test_field_initialization_performance(benchmark):
    """Benchmark field initialization performance."""
    def init_field():
        return PhaseField((100, 100, 100))
    
    result = benchmark(init_field)
    assert result.field.shape == (100, 100, 100)


def test_singularity_detection_performance(benchmark, large_field):
    """Benchmark singularity detection performance."""
    # Create a field with known singularities
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    z = np.linspace(-1, 1, 100)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    theta = np.arctan2(Y, X)
    large_field.field = np.exp(1j * theta)
    
    def find_singularities():
        large_field._find_singularities()
        return len(large_field.singularities)
    
    result = benchmark(find_singularities)
    assert result > 0


def test_field_update_performance(benchmark, large_field):
    """Benchmark field update performance."""
    transform = MobiusTransformation(1, 0, 0, 1)
    
    def update_field():
        large_field.update(transform)
        return large_field.field
    
    result = benchmark(update_field)
    assert result.shape == (100, 100, 100)


def test_phase_circulation_performance(benchmark, large_field):
    """Benchmark phase circulation calculation performance."""
    # Create a simple vortex field
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    z = np.linspace(-1, 1, 100)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    theta = np.arctan2(Y, X)
    large_field.field = np.exp(1j * theta)
    
    def calculate_circulation():
        return large_field._calculate_phase_circulation((50, 50, 50))
    
    result = benchmark(calculate_circulation)
    assert abs(abs(result) - 2*np.pi) < 0.1


def test_memory_usage(benchmark, large_field):
    """Benchmark memory usage during field operations."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    def measure_memory():
        initial_memory = process.memory_info().rss
        large_field._find_singularities()
        final_memory = process.memory_info().rss
        return final_memory - initial_memory
    
    result = benchmark(measure_memory)
    assert result > 0  # Memory usage should be positive 