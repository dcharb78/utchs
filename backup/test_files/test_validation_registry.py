"""Test suite for the validation registry module."""

import pytest
import numpy as np
from decimal import Decimal
from utchs.utils.validation_registry import validation_registry, UTCHSValidationError

@pytest.fixture
def registry():
    """Provide a validation registry instance for testing."""
    return validation_registry

def test_module_data_validation_position(registry):
    """Test validation of Position module data."""
    valid_data = {
        "number": 7,
        "phase": 1.5 + 2.3j,
        "energy_level": Decimal("100.5"),
        "spatial_location": [Decimal("1.0"), Decimal("2.0"), Decimal("3.0")],
        "velocity_vector": [Decimal("0.1"), Decimal("0.2"), Decimal("0.3")]
    }
    registry.validate_module_data("Position", valid_data)

def test_module_data_validation_phase_field(registry):
    """Test validation of PhaseField module data."""
    valid_data = {
        "field_data": np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=complex),
        "grid_size": (10, 10, 10),
        "singularities": [{"x": 1, "y": 2, "z": 3}]
    }
    registry.validate_module_data("PhaseField", valid_data)

def test_module_data_validation_vector_field(registry):
    """Test validation of VectorField module data."""
    valid_data = {
        "field_data": np.array([
            [Decimal("1.0"), Decimal("2.0"), Decimal("3.0")],
            [Decimal("4.0"), Decimal("5.0"), Decimal("6.0")]
        ]),
        "magnitude_range": [Decimal("0.0"), Decimal("10.0"), Decimal("20.0")]
    }
    registry.validate_module_data("VectorField", valid_data)

def test_invalid_module_data(registry):
    """Test validation fails for invalid module data."""
    invalid_data = {
        "number": 15,  # Out of range
        "phase": 1.5,  # Not complex
        "energy_level": -1,  # Negative energy
        "spatial_location": [1.0, 2.0],  # Wrong length
        "velocity_vector": "invalid"  # Wrong type
    }
    with pytest.raises(UTCHSValidationError):
        registry.validate_module_data("Position", invalid_data)

def test_complex_validation(registry):
    """Test complex number validation."""
    data = {
        "field_data": np.array([[1, 2], [3, 4]], dtype=float),  # Not complex
        "grid_size": (10, 10, 10),
        "singularities": []
    }
    with pytest.raises(UTCHSValidationError, match="must contain complex numbers"):
        registry.validate_module_data("PhaseField", data)

def test_decimal_array_validation(registry):
    """Test validation of decimal arrays."""
    # Test valid decimal array
    valid_data = {
        "field_data": np.array([[Decimal("1.0"), Decimal("2.0")], 
                              [Decimal("3.0"), Decimal("4.0")]])
    }
    registry.validate_module_data("test_module", valid_data)
    
    # Test invalid decimal array
    invalid_data = {
        "field_data": np.array([[1.0, 2.0], [3.0, 4.0]])
    }
    with pytest.raises(UTCHSValidationError, match="Invalid numeric value in field_data"):
        registry.validate_module_data("test_module", invalid_data)

def test_grid_size_validation(registry):
    """Test validation of grid size tuples."""
    data = {
        "field_data": np.array([[1+1j, 2+2j]], dtype=complex),
        "grid_size": (10, 10),  # Should be 3D
        "singularities": []
    }
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry.validate_module_data("PhaseField", data)
    assert "grid_size must be a tuple of length 3" in str(exc_info.value)

def test_array_shape_validation(registry):
    """Test validation of array shapes."""
    data = {
        "number": 7,
        "spatial_location": [Decimal("1.0"), Decimal("2.0")],  # Missing Z coordinate
        "phase": 1.5 + 2.3j,
        "energy_level": Decimal("100.5")
    }
    with pytest.raises(UTCHSValidationError, match="must have shape \\(3,\\)"):
        registry.validate_module_data("Position", data)

def test_singularity_validation(registry):
    """Test validation of singularity data."""
    data = {
        "field_data": np.array([[1+1j, 2+2j]], dtype=complex),
        "grid_size": (10, 10, 10),
        "singularities": [{"x": 1, "y": 2}]  # Missing z coordinate
    }
    with pytest.raises(UTCHSValidationError, match="must have x, y, and z coordinates"):
        registry.validate_module_data("PhaseField", data)

def test_magnitude_range_validation(registry):
    """Test validation of magnitude ranges."""
    data = {
        "field_data": np.array([
            [Decimal("1.0"), Decimal("2.0"), Decimal("3.0")],
            [Decimal("4.0"), Decimal("5.0"), Decimal("6.0")]
        ]),
        "magnitude_range": [Decimal("10.0"), Decimal("0.0"), Decimal("20.0")]  # Invalid range (not ascending)
    }
    with pytest.raises(UTCHSValidationError, match="must be in ascending order"):
        registry.validate_module_data("VectorField", data)

def test_energy_level_validation(registry):
    """Test validation of energy levels."""
    data = {
        "number": 7,
        "phase": 1.5 + 2.3j,
        "energy_level": Decimal("-50.5"),  # Negative energy
        "spatial_location": [Decimal("1.0"), Decimal("2.0"), Decimal("3.0")]
    }
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry.validate_module_data("Position", data)
    assert "energy_level must be >= 0" in str(exc_info.value)

def test_position_number_validation(registry):
    """Test validation of position numbers."""
    data = {
        "number": 0,  # Invalid position number (must be positive)
        "phase": 1.5 + 2.3j,
        "energy_level": Decimal("100.5"),
        "spatial_location": [Decimal("1.0"), Decimal("2.0"), Decimal("3.0")]
    }
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry.validate_module_data("Position", data)
    assert "number must be >= 1" in str(exc_info.value)

def test_complex_infinity_handling(registry):
    """Test handling of complex infinity values."""
    # Test valid complex value
    valid_data = {
        "field_data": complex(1.0, 2.0)
    }
    registry.validate_module_data("test_module", valid_data)
    
    # Test complex infinity
    invalid_data = {
        "field_data": complex(np.inf, 2.0)
    }
    with pytest.raises(UTCHSValidationError, match="Complex value in field_data cannot contain infinity"):
        registry.validate_module_data("test_module", invalid_data)

def test_mixed_type_array_handling(registry):
    """Test handling of mixed type arrays."""
    # Test valid mixed type array
    valid_data = {
        "field_data": np.array([[1.0, 2.0], [3.0, 4.0]])
    }
    registry.validate_module_data("test_module", valid_data)
    
    # Test invalid mixed type array
    invalid_data = {
        "field_data": np.array([[1.0, "2.0"], [3.0, 4.0]])
    }
    with pytest.raises(UTCHSValidationError, match="Invalid numeric value in field_data"):
        registry.validate_module_data("test_module", invalid_data)

def test_high_precision_decimal_overflow(registry):
    """Test handling of high precision decimal overflow."""
    # Test valid precision
    valid_data = {
        "field_data": Decimal("1.23456789")
    }
    registry.validate_module_data("test_module", valid_data)
    
    # Test precision overflow
    invalid_data = {
        "field_data": Decimal("1.2345678901234567890123456789")
    }
    with pytest.raises(UTCHSValidationError, match="Decimal precision exceeds maximum"):
        registry.validate_module_data("test_module", invalid_data)

def test_nested_array_shape_consistency(registry):
    """Test validation of nested arrays with inconsistent shapes."""
    data = {
        "field_data": [
            [Decimal("1.0"), Decimal("2.0")],
            [Decimal("3.0"), Decimal("4.0"), Decimal("5.0")]  # Inconsistent length
        ],
        "magnitude_range": [Decimal("0.0"), Decimal("10.0")]
    }
    with pytest.raises(ValueError, match="inhomogeneous shape"):
        registry.validate_module_data("VectorField", data)

def test_memory_cleanup_large_arrays(registry):
    """Test memory cleanup with large array operations."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Create and validate a large array
    large_array = np.array([Decimal(str(i % 1000)) for i in range(100000)])  # Reduced size, reuse values
    data = {
        "field_data": large_array,
        "magnitude_range": [Decimal("0.0"), Decimal("1000.0")]
    }
    
    registry.validate_module_data("VectorField", data)
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Check memory usage
    final_memory = process.memory_info().rss
    memory_diff = final_memory - initial_memory
    
    # Memory difference should be reasonable (less than 200MB)
    assert memory_diff < 200 * 1024 * 1024, f"Memory not properly cleaned up: {memory_diff / (1024*1024):.2f}MB used"

def test_position_validation(registry):
    """Test validation of position data."""
    # Test valid position data
    valid_data = {
        "number": 1,
        "spatial_location": (Decimal("1.0"), Decimal("2.0"), Decimal("3.0")),
        "velocity_vector": (Decimal("0.1"), Decimal("0.2"), Decimal("0.3"))
    }
    registry.validate_module_data("Position", valid_data)
    
    # Test invalid position data
    invalid_data = {
        "number": 1,
        "spatial_location": (1.0, 2.0, 3.0),
        "velocity_vector": (0.1, 0.2, 0.3)
    }
    with pytest.raises(UTCHSValidationError, match="Invalid numeric value in spatial_location"):
        registry.validate_module_data("Position", invalid_data)

def test_phase_field_validation(registry):
    """Test validation of phase field data."""
    # Test valid phase field data
    valid_data = {
        "field_data": np.array([[complex(1.0, 2.0), complex(3.0, 4.0)],
                               [complex(5.0, 6.0), complex(7.0, 8.0)]], dtype=np.complex128)
    }
    registry.validate_module_data("PhaseField", valid_data)
    
    # Test invalid phase field data
    invalid_data = {
        "field_data": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    }
    with pytest.raises(UTCHSValidationError, match="Field field_data must be a complex array"):
        registry.validate_module_data("PhaseField", invalid_data)

def test_energy_field_validation(registry):
    """Test validation of EnergyField module data."""
    # Valid energy field data
    valid_data = {
        "field_data": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "grid_spacing": 0.1
    }
    
    # This should not raise any exceptions
    registry.validate_module_data("EnergyField", valid_data)
    
    # Test invalid grid spacing
    invalid_data = valid_data.copy()
    invalid_data["grid_spacing"] = -0.1
    with pytest.raises(UTCHSValidationError, match="must be >= 0"):
        registry.validate_module_data("EnergyField", invalid_data)
        
    # Test invalid field data type
    invalid_data = valid_data.copy()
    invalid_data["field_data"] = "not an array"
    with pytest.raises(UTCHSValidationError):
        registry.validate_module_data("EnergyField", invalid_data)

def test_vector_field_validation(registry):
    """Test validation of VectorField module data."""
    # Valid vector field data
    valid_data = {
        "field_data": np.array([
            [Decimal("1.0"), Decimal("2.0")],
            [Decimal("3.0"), Decimal("4.0")]
        ]),
        "magnitude_range": [Decimal("0.0"), Decimal("5.0")]
    }
    
    # This should not raise any exceptions
    registry.validate_module_data("VectorField", valid_data)
    
    # Test invalid magnitude range
    invalid_data = valid_data.copy()
    invalid_data["magnitude_range"] = [-1, 5]  # Negative value
    with pytest.raises(UTCHSValidationError, match="Array validation failed for magnitude_range"):
        registry.validate_module_data("VectorField", invalid_data)
        
    # Test mixed types in field data
    invalid_data = valid_data.copy()
    invalid_data["field_data"] = [[1, "2"], [3, 4]]  # Mixed numeric and string
    with pytest.raises(UTCHSValidationError):
        registry.validate_module_data("VectorField", invalid_data)

def test_numeric_precision(registry):
    """Test numeric precision validation."""
    # Test valid precision
    valid_data = {
        "field_data": Decimal("1.23456789")
    }
    registry.validate_module_data("test_module", valid_data)
    
    # Test precision overflow
    invalid_data = {
        "field_data": Decimal("1.2345678901234567890123456789")
    }
    with pytest.raises(UTCHSValidationError, match="Decimal precision exceeds maximum"):
        registry.validate_module_data("test_module", invalid_data)

def test_array_validation(registry):
    """Test array validation."""
    # Test valid array conversion
    arr = np.array([1.0, 2.0, 3.0])
    result = registry._validate_array_dtype(
        arr,
        np.float64,
        "test_array"
    )
    assert result.dtype == np.float64
    
    # Test invalid conversion
    with pytest.raises(UTCHSValidationError, match="Could not convert test_array to array"):
        registry._validate_array_dtype(
            ["not", "a", "number"],
            np.float64,
            "test_array"
        )
        
    # Test complex array conversion
    arr = np.array([1, 2+3j, 4])
    result = registry._validate_array_dtype(
        arr,
        complex,
        "test_array"
    )
    assert np.issubdtype(result.dtype, np.complexfloating)

def test_method_name_validation(registry):
    """Test method name validation."""
    # Test valid method names
    registry._validate_method_name("calculate_energy", "calculate")
    registry._validate_method_name("update_position", "update")
    registry._validate_method_name("get_phase", "get")
    registry._validate_method_name("find_singularities", "find")
    
    # Test invalid method names
    with pytest.raises(UTCHSValidationError):
        registry._validate_method_name("invalidMethod", "calculate")
    with pytest.raises(UTCHSValidationError):
        registry._validate_method_name("Update_state", "update")  # Wrong case 