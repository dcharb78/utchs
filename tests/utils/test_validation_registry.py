"""Tests for the validation registry module."""

import pytest
import numpy as np
from decimal import Decimal, getcontext
from typing import Dict, List, Optional

from utchs.utils.validation_registry import ValidationRegistry, UTCHSValidationError
from utchs.core.position import Position
from utchs.core.cycle import Cycle
from utchs.core.structure import ScaledStructure
from utchs.core.torus import Torus
from utchs.math.phase_field import PhaseField
from utchs.fields.energy_field import EnergyField

@pytest.fixture
def registry():
    """Create a fresh validation registry for each test."""
    return ValidationRegistry()

class TestCoreConceptValidation:
    """Test validation of core UTCHS concepts."""
    
    def test_valid_core_concepts(self, registry):
        """Test that valid core concepts pass validation."""
        valid_concepts = ["Position", "Cycle", "ScaledStructure", "Torus"]
        for concept in valid_concepts:
            registry._validate_core_concept(concept)
            
    def test_invalid_core_concept(self, registry):
        """Test that invalid core concepts raise ValidationError."""
        with pytest.raises(UTCHSValidationError) as exc:
            registry._validate_core_concept("InvalidConcept")
        assert "not a recognized UTCHS core concept" in str(exc.value)
            
class TestFieldTypeValidation:
    """Test validation of UTCHS field types."""
    
    def test_valid_field_types(self, registry):
        """Test that valid field types pass validation."""
        valid_types = ["PhaseField", "EnergyField", "TorsionField"]
        for field_type in valid_types:
            registry._validate_field_type(field_type)
            
    def test_invalid_field_type(self, registry):
        """Test that invalid field types raise ValidationError."""
        with pytest.raises(UTCHSValidationError) as exc:
            registry._validate_field_type("InvalidField")
        assert "not a recognized UTCHS field type" in str(exc.value)
            
class TestAttributeNameValidation:
    """Test validation of attribute names."""
    
    @pytest.mark.parametrize("pattern_type,valid_names", [
        ("number", ["1", "2", "13"]),
        ("id", ["test_id", "position1", "cycle_A"]),
        ("field", ["phase_field", "energy_field"]),
        ("grid_size", ["grid_size"]),
        ("spatial_location", ["spatial_location"]),
        ("rotational_angle", ["rotational_angle"]),
        ("energy_level", ["energy_level"]),
        ("phase", ["phase"])
    ])
    def test_valid_attribute_names(self, registry, pattern_type, valid_names):
        """Test that valid attribute names pass validation."""
        for name in valid_names:
            registry._validate_attribute_name(name, pattern_type)
            
    @pytest.mark.parametrize("pattern_type,invalid_name", [
        ("number", "abc"),
        ("id", "invalid@id"),
        ("field", "invalid"),
        ("grid_size", "size"),
        ("spatial_location", "location"),
        ("rotational_angle", "angle"),
        ("energy_level", "energy"),
        ("phase", "phase_value")
    ])
    def test_invalid_attribute_names(self, registry, pattern_type, invalid_name):
        """Test that invalid attribute names raise ValidationError."""
        with pytest.raises(UTCHSValidationError) as exc:
            registry._validate_attribute_name(invalid_name, pattern_type)
        assert "does not follow the" in str(exc.value)
            
class TestMethodNameValidation:
    """Test validation of method names."""
    
    @pytest.mark.parametrize("pattern_type,valid_names", [
        ("calculate", ["calculate_energy", "calculate_phase_difference"]),
        ("update", ["update_position", "update_field_values"]),
        ("get", ["get_current_position", "get_energy_level"]),
        ("find", ["find_resonance", "find_nearest_position"])
    ])
    def test_valid_method_names(self, registry, pattern_type, valid_names):
        """Test that valid method names pass validation."""
        for name in valid_names:
            registry._validate_method_name(name, pattern_type)
            
    @pytest.mark.parametrize("pattern_type,invalid_name", [
        ("calculate", "calculateEnergy"),
        ("update", "Update_position"),
        ("get", "getCurrent"),
        ("find", "Find")
    ])
    def test_invalid_method_names(self, registry, pattern_type, invalid_name):
        """Test that invalid method names raise ValidationError."""
        with pytest.raises(UTCHSValidationError) as exc:
            registry._validate_method_name(invalid_name, pattern_type)
        assert "does not follow the" in str(exc.value)
            
class TestPositionNumberValidation:
    """Test validation of position numbers."""
    
    def test_valid_position_numbers(self, registry):
        """Test that valid position numbers pass validation."""
        for number in range(1, 14):
            registry._validate_position_number(number)
            
    @pytest.mark.parametrize("invalid_number", [0, 14, -1, 1.5, "1"])
    def test_invalid_position_numbers(self, registry, invalid_number):
        """Test that invalid position numbers raise ValidationError."""
        with pytest.raises(UTCHSValidationError) as exc:
            registry._validate_position_number(invalid_number)
        assert "must be an integer between 1 and 13" in str(exc.value)
            
class TestGridDimensionsValidation:
    """Test validation of grid dimensions."""
    
    def test_valid_grid_dimensions(self, registry):
        """Test that valid grid dimensions pass validation."""
        valid_dimensions = [
            (1, 1, 1),
            (10, 10, 10),
            (5, 8, 3)
        ]
        for dimensions in valid_dimensions:
            registry._validate_grid_dimensions(dimensions)
            
    @pytest.mark.parametrize("invalid_dimensions", [
        (0, 1, 1),
        (1, -1, 1),
        (1.5, 2, 3),
        (1, 1),
        [1, 1, 1],
        "invalid"
    ])
    def test_invalid_grid_dimensions(self, registry, invalid_dimensions):
        """Test that invalid grid dimensions raise ValidationError."""
        with pytest.raises(UTCHSValidationError) as exc:
            registry._validate_grid_dimensions(invalid_dimensions)
        assert "Grid dimensions must be" in str(exc.value)
            
def test_validator_registration(registry):
    """Test that validators can be registered and listed."""
    expected_validators = {
        "type", "range", "array_shape", "array_dtype",
        "list_length", "set_membership", "model", "callable",
        "path", "dict_keys", "core_concept", "field_type",
        "attribute_name", "method_name", "position_number",
        "grid_dimensions"
    }
    assert registry.list_validators() == expected_validators 

class TestNumericPrecisionValidation:
    """Test validation of numeric precision requirements."""
    
    def test_metacycle_number(self, registry):
        """Test validation of metacycle numbers."""
        # Valid cases
        assert registry._validate_numeric_precision(1, "metacycle_number") == 1
        assert registry._validate_numeric_precision(999, "metacycle_number") == 999
        
        # Invalid cases
        with pytest.raises(UTCHSValidationError, match="exceeds maximum allowed digits"):
            registry._validate_numeric_precision(1000, "metacycle_number")  # Too many digits
        with pytest.raises(UTCHSValidationError, match="must be of type int"):
            registry._validate_numeric_precision(1.5, "metacycle_number")  # Not an integer
            
    def test_vector_decimal(self, registry):
        """Test validation of vector decimals."""
        # Valid cases
        value = Decimal("1." + "1" * 999999)  # Test high precision
        result = registry._validate_numeric_precision(value, "vector_decimal")
        assert isinstance(result, Decimal)
        assert result == value  # Check precision maintained
        
        # Test conversion from string
        result = registry._validate_numeric_precision("1.23456789", "vector_decimal")
        assert isinstance(result, Decimal)
        assert result == Decimal("1.23456789")
        
        # Invalid cases
        with pytest.raises(UTCHSValidationError):
            registry._validate_numeric_precision("invalid", "vector_decimal")
            
    def test_phase_complex(self, registry):
        """Test validation of complex phase values."""
        # Valid cases
        value = 1+2j
        result = registry._validate_numeric_precision(value, "phase_complex")
        assert isinstance(result, (complex, np.complex128))
        assert result == value
        
        value = np.complex128(1+2j)
        result = registry._validate_numeric_precision(value, "phase_complex")
        assert isinstance(result, (complex, np.complex128))
        assert result == value
        
        # Invalid cases
        with pytest.raises(UTCHSValidationError, match="must be a complex number"):
            registry._validate_numeric_precision(1.0, "phase_complex")  # Not complex
            
    def test_numpy_arrays(self, registry):
        """Test validation of numpy array types."""
        # Valid cases
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = registry._validate_numeric_precision(arr, "numpy_standard")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, arr)
        
        # Test conversion
        arr = [1.0, 2.0, 3.0]  # List
        result = registry._validate_numeric_precision(arr, "numpy_standard")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.array(arr, dtype=np.float64))
        
        # Invalid cases
        with pytest.raises(UTCHSValidationError, match="Could not convert"):
            registry._validate_numeric_precision([1, "invalid"], "numpy_standard")
            
    def test_coordinate_precision(self, registry):
        """Test validation of coordinate values."""
        # Valid cases
        coords = [
            Decimal("123.456789123456789"),
            "987.654321987654321",
            123.456  # Should convert to Decimal
        ]
        for coord in coords:
            result = registry._validate_numeric_precision(coord, "coordinate_decimal")
            assert isinstance(result, Decimal)
            if isinstance(coord, Decimal):
                assert result == coord
            else:
                assert result == Decimal(str(coord))
            
        # Test precision maintenance
        value = Decimal("1." + "9" * 999999)
        result = registry._validate_numeric_precision(value, "coordinate_decimal")
        assert result == value  # Should maintain full precision
        
        # Invalid cases
        with pytest.raises(UTCHSValidationError):
            registry._validate_numeric_precision("invalid", "coordinate_decimal")
            
class TestModuleDataValidation:
    """Test validation of module-specific data formats."""
    
    def test_position_data(self, registry):
        """Test validation of Position module data."""
        valid_data = {
            "number": 7,
            "phase": 1+2j,
            "energy_level": Decimal("123.456"),
            "spatial_location": Decimal("1.23456"),
            "velocity_vector": Decimal("9.87654")
        }
        registry.validate_module_data("Position", valid_data)
        
        # Test invalid number
        invalid_data = valid_data.copy()
        invalid_data["number"] = 14
        with pytest.raises(UTCHSValidationError):
            registry.validate_module_data("Position", invalid_data)
            
        # Test invalid phase
        invalid_data = valid_data.copy()
        invalid_data["phase"] = 1.0  # Not complex
        with pytest.raises(UTCHSValidationError, match="must be a complex number"):
            registry.validate_module_data("Position", invalid_data)
            
    def test_phase_field_data(self, registry):
        """Test validation of PhaseField module data."""
        valid_data = {
            "field_data": np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128),
            "grid_size": (2, 2, 1),
            "singularities": [{"position": (1, 1, 1), "charge": 1}]
        }
        registry.validate_module_data("PhaseField", valid_data)
        
        # Test invalid field data type
        invalid_data = valid_data.copy()
        invalid_data["field_data"] = np.array([[1, 2], [3, 4]], dtype=np.float64)  # Not complex
        with pytest.raises(UTCHSValidationError, match="Could not convert"):
            registry.validate_module_data("PhaseField", invalid_data)
            
    def test_vector_field_data(self, registry):
        """Test validation of VectorField module data."""
        valid_data = {
            "field_data": Decimal("1.23456789"),
            "magnitude_range": Decimal("9.87654321")
        }
        registry.validate_module_data("VectorField", valid_data)
        
        # Test precision requirements
        high_precision = Decimal("1." + "9" * 999999)
        valid_data["field_data"] = high_precision
        registry.validate_module_data("VectorField", valid_data)
        
        # Test invalid data type
        invalid_data = valid_data.copy()
        invalid_data["field_data"] = 1.23  # Not Decimal
        with pytest.raises(UTCHSValidationError):
            registry.validate_module_data("VectorField", invalid_data) 