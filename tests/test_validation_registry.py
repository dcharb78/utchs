"""Test suite for the validation registry module."""

import pytest
import numpy as np
import re
from decimal import Decimal
from typing import Dict, Any, List
from pydantic import BaseModel
from utchs.utils.validation_registry import validation_registry, UTCHSValidationError
from utchs.utils.validation_registry import CORE_CONCEPTS, FIELD_TYPES, ATTRIBUTE_PATTERNS, METHOD_PATTERNS

@pytest.fixture
def registry():
    """Provide a validation registry instance for testing."""
    return validation_registry

def test_basic_validation_capabilities(registry):
    """Test the basic validation capabilities of the registry."""
    # Test _validate_type
    registry._validate_type(5, int, "test_int")
    with pytest.raises(UTCHSValidationError):
        registry._validate_type("5", int, "test_int")
    
    # Test _validate_range
    registry._validate_range(5, 1, 10, "test_range")
    with pytest.raises(UTCHSValidationError):
        registry._validate_range(0, 1, 10, "test_range")
    with pytest.raises(UTCHSValidationError):
        registry._validate_range(11, 1, 10, "test_range")
    
    # Test _validate_array_shape
    arr = np.array([[1, 2], [3, 4]])
    registry._validate_array_shape(arr, (2, 2), "test_array")
    with pytest.raises(UTCHSValidationError):
        registry._validate_array_shape(arr, (3, 2), "test_array")

def test_validate_module_data_position(registry):
    """Test validation of Position module data."""
    valid_data = {
        "number": 7,
        "phase": 1.5 + 2.3j,
        "energy_level": Decimal("100.5"),
        "spatial_location": [Decimal("1.0"), Decimal("2.0"), Decimal("3.0")],
        "velocity_vector": [Decimal("0.1"), Decimal("0.2"), Decimal("0.3")]
    }
    # Should not raise an exception
    registry.validate_module_data("Position", valid_data)
    
    # Test invalid number
    invalid_data = valid_data.copy()
    invalid_data["number"] = 0
    with pytest.raises(UTCHSValidationError):
        registry.validate_module_data("Position", invalid_data)
    
    # Test invalid phase
    invalid_data = valid_data.copy()
    invalid_data["phase"] = 1.5  # Not complex
    with pytest.raises(UTCHSValidationError):
        registry.validate_module_data("Position", invalid_data)
    
    # Test invalid spatial_location length
    invalid_data = valid_data.copy()
    invalid_data["spatial_location"] = [Decimal("1.0"), Decimal("2.0")]  # Missing third coordinate
    with pytest.raises(UTCHSValidationError):
        registry.validate_module_data("Position", invalid_data)

def test_validate_module_data_vector_field(registry):
    """Test validation of VectorField module data."""
    valid_data = {
        "field_data": np.array([
            [Decimal("1.0"), Decimal("2.0"), Decimal("3.0")],
            [Decimal("4.0"), Decimal("5.0"), Decimal("6.0")]
        ]),
        "magnitude_range": [Decimal("0.0"), Decimal("10.0"), Decimal("20.0")]
    }
    # Should not raise an exception
    registry.validate_module_data("VectorField", valid_data)
    
    # Test invalid magnitude_range (not ascending)
    invalid_data = valid_data.copy()
    invalid_data["magnitude_range"] = [Decimal("10.0"), Decimal("0.0"), Decimal("20.0")]
    with pytest.raises(UTCHSValidationError):
        registry.validate_module_data("VectorField", invalid_data)
    
    # Test missing required field
    invalid_data = {"field_data": valid_data["field_data"]}  # Missing magnitude_range
    with pytest.raises(UTCHSValidationError):
        registry.validate_module_data("VectorField", invalid_data)

def test_module_specific_validation(registry):
    """Test module-specific validation rules."""
    # Test PhaseField validation (must be complex array)
    valid_phase_data = {
        "field_data": np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=complex),
        "grid_size": (10, 10, 10),
        "singularities": [{"x": 1, "y": 2, "z": 3}]
    }
    registry.validate_module_data("PhaseField", valid_phase_data)
    
    invalid_phase_data = valid_phase_data.copy()
    invalid_phase_data["field_data"] = np.array([[1, 2], [3, 4]], dtype=float)
    with pytest.raises(UTCHSValidationError):
        registry.validate_module_data("PhaseField", invalid_phase_data)
    
    # Test EnergyField validation (grid_spacing must be positive)
    valid_energy_data = {
        "field_data": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "grid_spacing": 0.1
    }
    registry.validate_module_data("EnergyField", valid_energy_data)
    
    invalid_energy_data = valid_energy_data.copy()
    invalid_energy_data["grid_spacing"] = -0.1
    with pytest.raises(UTCHSValidationError):
        registry.validate_module_data("EnergyField", invalid_energy_data)

def test_list_length_validation(registry):
    """Test validation of list lengths."""
    # Test valid list length
    valid_list = [1, 2, 3, 4, 5]
    registry._validate_list_length(valid_list, min_length=3, max_length=10, name="test_list")
    
    # Test list too short
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry._validate_list_length([1], min_length=2, name="short_list")
    assert "must have length >= 2" in str(exc_info.value)
    
    # Test list too long
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry._validate_list_length([1, 2, 3], max_length=2, name="long_list")
    assert "must have length <= 2" in str(exc_info.value)

def test_set_membership_validation(registry):
    """Test validation of set membership."""
    # Valid membership
    allowed_values = {"apple", "banana", "cherry"}
    registry._validate_set_membership("apple", allowed_values, name="fruit")
    
    # Invalid membership
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry._validate_set_membership("orange", allowed_values, name="fruit")
    assert "must be one of" in str(exc_info.value)
    assert "apple" in str(exc_info.value)
    assert "orange" in str(exc_info.value)

def test_model_validation(registry):
    """Test validation using Pydantic models."""
    # Define a simple Pydantic model for testing
    class TestModel(BaseModel):
        name: str
        age: int
        
    # Valid data
    valid_data = {"name": "Test User", "age": 25}
    result = registry._validate_model(valid_data, TestModel, name="test_model")
    assert isinstance(result, TestModel)
    assert result.name == "Test User"
    assert result.age == 25
    
    # Invalid data (wrong type)
    invalid_data = {"name": "Test User", "age": "twenty-five"}
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry._validate_model(invalid_data, TestModel, name="test_model")
    assert "Invalid test_model" in str(exc_info.value)
    
    # Invalid data (missing field)
    invalid_data = {"name": "Test User"}
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry._validate_model(invalid_data, TestModel, name="test_model")
    assert "Invalid test_model" in str(exc_info.value)

def test_core_concept_validation(registry):
    """Test validation of core UTCHS concepts."""
    # Test valid core concepts
    for concept in CORE_CONCEPTS.keys():
        registry._validate_core_concept(concept)
    
    # Test invalid core concept
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry._validate_core_concept("InvalidConcept")
    assert "not a recognized UTCHS core concept" in str(exc_info.value)
    for concept in CORE_CONCEPTS.keys():
        assert concept in str(exc_info.value)

def test_field_type_validation(registry):
    """Test validation of UTCHS field types."""
    # Test valid field types
    for field_type in FIELD_TYPES.keys():
        registry._validate_field_type(field_type)
    
    # Test invalid field type
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry._validate_field_type("InvalidFieldType")
    assert "not a recognized UTCHS field type" in str(exc_info.value)
    for field_type in FIELD_TYPES.keys():
        assert field_type in str(exc_info.value)

def test_attribute_name_validation(registry):
    """Test validation of attribute names against patterns."""
    # Test valid attribute names
    for pattern_type, pattern in ATTRIBUTE_PATTERNS.items():
        # Create an example name that matches the pattern
        if pattern_type == "number":
            example_name = "123"
        elif pattern_type == "id":
            example_name = "test_id_123"
        elif pattern_type == "field":
            example_name = "test_field"
        elif pattern_type == "grid_size":
            example_name = "grid_size"
        elif pattern_type == "spatial_location":
            example_name = "spatial_location"
        elif pattern_type == "rotational_angle":
            example_name = "rotational_angle"
        elif pattern_type == "energy_level":
            example_name = "energy_level"
        elif pattern_type == "phase":
            example_name = "phase"
        else:
            # Skip if we don't have a good example
            continue
            
        if re.match(pattern, example_name):
            registry._validate_attribute_name(example_name, pattern_type)
    
    # Test invalid attribute name
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry._validate_attribute_name("invalid@name", "id")
    assert "does not follow the id pattern" in str(exc_info.value)
    
    # Test unknown pattern type
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry._validate_attribute_name("name", "unknown_pattern")
    assert "Unknown attribute pattern type" in str(exc_info.value)

def test_method_name_validation(registry):
    """Test validation of method names against patterns."""
    # Test valid method names
    for pattern_type, pattern in METHOD_PATTERNS.items():
        # Create an example method name that matches the pattern
        if pattern_type == "calculate":
            example_name = "calculate_value"
        elif pattern_type == "update":
            example_name = "update_position"
        elif pattern_type == "get":
            example_name = "get_value"
        elif pattern_type == "find":
            example_name = "find_element"
        else:
            # Skip if we don't have a good example
            continue
            
        if re.match(pattern, example_name):
            registry._validate_method_name(example_name, pattern_type)
    
    # Test invalid method name
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry._validate_method_name("calculateValue", "calculate")
    assert "does not follow the calculate pattern" in str(exc_info.value)
    
    # Test unknown pattern type
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry._validate_method_name("method_name", "unknown_pattern")
    assert "Unknown method pattern type" in str(exc_info.value)

def test_dict_keys_validation(registry):
    """Test validation of dictionary keys."""
    # Test dictionary with required keys
    valid_dict = {"name": "Test", "age": 25, "city": "Example"}
    required_keys = {"name", "age"}
    registry._validate_dict_keys(valid_dict, required_keys=required_keys)
    
    # Test dictionary with missing required keys
    invalid_dict = {"name": "Test"}
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry._validate_dict_keys(invalid_dict, required_keys={"name", "age"})
    assert "is missing required keys" in str(exc_info.value)
    assert "age" in str(exc_info.value)
    
    # Test dictionary with allowed keys
    valid_dict = {"name": "Test", "age": 25}
    allowed_keys = {"name", "age", "city"}
    registry._validate_dict_keys(valid_dict, allowed_keys=allowed_keys)
    
    # Test dictionary with invalid keys
    invalid_dict = {"name": "Test", "age": 25, "invalid_key": "value"}
    with pytest.raises(UTCHSValidationError) as exc_info:
        registry._validate_dict_keys(invalid_dict, allowed_keys={"name", "age"})
    assert "contains invalid keys" in str(exc_info.value)
    assert "invalid_key" in str(exc_info.value)
    
    # Test dictionary with both required and allowed keys
    valid_dict = {"name": "Test", "age": 25}
    required_keys = {"name"}
    allowed_keys = {"name", "age", "city"}
    registry._validate_dict_keys(valid_dict, required_keys=required_keys, allowed_keys=allowed_keys) 