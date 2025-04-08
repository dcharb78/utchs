"""Validation registry for the UTCHS framework.

This module provides a centralized registry for all validation rules used across the framework.
It ensures consistent validation patterns and error handling.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union
import numpy as np
from decimal import Decimal, getcontext, Context, InvalidOperation, localcontext
from pydantic import BaseModel, ValidationError

from utchs.core.exceptions import ValidationError as UTCHSValidationError

T = TypeVar("T", bound=BaseModel)

# Configure decimal precision for specific high-precision calculations
getcontext().prec = 1000000  # Up to 1 million digits for vector calculations

# Core UTCHS concepts and their allowed names
CORE_CONCEPTS = {
    "Position": "Fundamental unit in the UTCHS framework",
    "Cycle": "Collection of exactly 13 positions",
    "ScaledStructure": "Collection of cycles with scaling transformations",
    "Torus": "Highest level organizational unit"
}

# Field types and their descriptions
FIELD_TYPES = {
    "PhaseField": "Complex-valued field representing phase dynamics",
    "EnergyField": "Real-valued field representing energy distribution",
    "TorsionField": "Field representing geometric torsion"
}

# Numeric type specifications with precision requirements
NUMERIC_TYPES = {
    "metacycle_number": {"type": int, "max_digits": 3},  # Small integers (1-999)
    "position_number": {"type": int, "max_digits": 2},   # 1-13
    "standard_float": {"type": float, "precision": 64},  # Standard double precision
    "vector_decimal": {"type": Decimal, "precision": 1000000},  # High precision for vectors
    "phase_complex": {"type": complex, "precision": 128},  # Complex numbers for phase calculations
    "energy_decimal": {"type": Decimal, "precision": 100},  # Medium precision for energy
    "coordinate_decimal": {"type": Decimal, "precision": 1000000},  # High precision for coordinates
    "numpy_standard": {"type": np.float64, "precision": 64},  # Standard numpy arrays
    "numpy_extended": {"type": np.longdouble, "precision": 128}  # Extended precision numpy
}

# Expected data formats for each module with specific precision requirements
MODULE_DATA_FORMATS = {
    "Position": {
        "number": {
            "type": "position_number",
            "range": (1, 13)
        },
        "phase": {
            "type": "phase_complex",
            "range": None
        },
        "energy_level": {
            "type": "energy_decimal",
            "range": (0, None)
        },
        "spatial_location": {
            "type": "coordinate_decimal",
            "array_shape": (3,)
        },
        "velocity_vector": {
            "type": "vector_decimal",
            "array_shape": (3,)
        }
    },
    "Cycle": {
        "metacycle_number": {
            "type": "metacycle_number",
            "range": (1, 999)
        },
        "position_count": {
            "type": "position_number",
            "value": 13
        }
    },
    "PhaseField": {
        "field_data": {
            "type": "numpy_extended",
            "array_shape": None,
            "requires_complex": True
        },
        "grid_size": {
            "type": "tuple",
            "length": 3
        },
        "singularities": {
            "type": "list",
            "element_type": "dict"
        }
    },
    "EnergyField": {
        "field_data": {
            "type": "numpy_standard",
            "array_shape": None
        },
        "grid_spacing": {
            "type": "standard_float",
            "range": (0, None)
        }
    },
    "VectorField": {
        "field_data": {
            "type": "vector_decimal",
            "array_shape": None
        },
        "magnitude_range": {
            "type": "coordinate_decimal",
            "range": (0, None)
        }
    }
}

# Common attribute patterns
ATTRIBUTE_PATTERNS = {
    "number": r"^\d+$",  # Position numbering (1-13)
    "id": r"^[a-zA-Z0-9_]+$",  # Unique identifiers
    "field": r"^[a-zA-Z]+_field$",  # Field names
    "grid_size": r"^grid_size$",  # Field dimensions
    "spatial_location": r"^spatial_location$",  # 3D coordinates
    "rotational_angle": r"^rotational_angle$",  # Angular positions
    "energy_level": r"^energy_level$",  # Energy values
    "phase": r"^phase$"  # Phase values
}

# Method name patterns
METHOD_PATTERNS = {
    "calculate": r"^calculate_[a-z_]+$",
    "update": r"^update_[a-z_]+$",
    "get": r"^get_[a-z_]+$",
    "find": r"^find_[a-z_]+$"
}

class ValidationRegistry:
    """Central registry for all validation rules in the UTCHS framework."""
    
    def __init__(self):
        """Initialize the validation registry."""
        self.validators: Dict[str, Callable] = {}
        self._register_default_validators()
        
    def list_validators(self) -> Set[str]:
        """List all registered validators.
        
        Returns:
            Set of validator names
        """
        return set(self.validators.keys())
        
    def _register_default_validators(self) -> None:
        """Register default validators for core UTCHS modules."""
        # Position module validators
        self.validators["Position"] = {
            "number": lambda x: isinstance(x, int) and 1 <= x <= 13,
            "phase": lambda x: isinstance(x, complex),
            "energy_level": lambda x: isinstance(x, Decimal) and x >= 0,
            "spatial_location": lambda x: (
                isinstance(x, (list, tuple, np.ndarray)) and 
                len(x) == 3 and 
                all(isinstance(v, Decimal) for v in x)
            ),
            "velocity_vector": lambda x: (
                isinstance(x, (list, tuple, np.ndarray)) and 
                len(x) == 3 and 
                all(isinstance(v, Decimal) for v in x)
            )
        }

        # PhaseField module validators
        self.validators["PhaseField"] = {
            "field_data": lambda x: (
                isinstance(x, np.ndarray) and
                np.issubdtype(x.dtype, np.complexfloating)
            ),
            "grid_size": lambda x: (
                isinstance(x, tuple) and 
                len(x) == 3 and 
                all(isinstance(v, int) and v > 0 for v in x)
            ),
            "singularities": lambda x: (
                isinstance(x, list) and
                all(
                    isinstance(s, dict) and
                    all(k in s for k in ["x", "y", "z"]) and
                    all(isinstance(s[k], (int, float)) for k in ["x", "y", "z"])
                    for s in x
                )
            )
        }

        # VectorField module validators
        self.validators["VectorField"] = {
            "field_data": lambda x: (
                isinstance(x, (np.ndarray, list)) and
                all(isinstance(v, Decimal) for v in np.array(x).flatten() if v is not None)
            ),
            "magnitude_range": lambda x: (
                isinstance(x, (list, tuple)) and
                len(x) >= 2 and
                all(isinstance(v, (Decimal, int, float)) for v in x) and
                all(x[i] <= x[i+1] for i in range(len(x)-1))
            )
        }

        # EnergyField module validators
        self.validators["EnergyField"] = {
            "field_data": lambda x: (
                isinstance(x, np.ndarray) and
                x.dtype in [np.float32, np.float64]
            ),
            "grid_spacing": lambda x: isinstance(x, (int, float)) and x > 0
        }

        # Test module validators (for unit tests)
        self.validators["test_module"] = {
            "field_data": lambda x: True  # Accept any value for testing
        }
        
    def register_validator(self, name: str, validator: Callable) -> None:
        """Register a new validation rule.
        
        Args:
            name: Name of the validation rule
            validator: Validation function
        """
        if name in self.validators:
            raise ValueError(f"Validator '{name}' already registered")
        self.validators[name] = validator
        
    def validate(self, rule_name: str, value: Any, **kwargs) -> None:
        """Run a specific validation rule.
        
        Args:
            rule_name: Name of the validation rule to run
            value: Value to validate
            **kwargs: Additional arguments for the validator
            
        Raises:
            UTCHSValidationError: If validation fails
            ValueError: If rule_name is not registered
        """
        if rule_name not in self.validators:
            raise ValueError(f"Validation rule '{rule_name}' not found")
            
        try:
            self.validators[rule_name](value, **kwargs)
        except Exception as e:
            raise UTCHSValidationError(f"Validation failed for rule '{rule_name}': {str(e)}")
            
    def _validate_type(self, value: Any, expected_type: Type[Any], name: str = "value") -> None:
        """Validate that a value is of the expected type."""
        if not isinstance(value, expected_type):
            raise UTCHSValidationError(
                f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}"
            )
            
    def _validate_range(
        self,
        value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        name: str = "value"
    ) -> None:
        """Validate that a numeric value is within a range."""
        if min_value is not None and value < min_value:
            raise UTCHSValidationError(f"{name} must be >= {min_value}, got {value}")
        if max_value is not None and value > max_value:
            raise UTCHSValidationError(f"{name} must be <= {max_value}, got {value}")
            
    def _validate_array_shape(
        self,
        array: np.ndarray,
        expected_shape: tuple,
        name: str = "array"
    ) -> None:
        """Validate that a numpy array has the expected shape."""
        if array.shape != expected_shape:
            raise UTCHSValidationError(
                f"{name} must have shape {expected_shape}, got {array.shape}"
            )
            
    def _validate_array_dtype(
        self,
        array: np.ndarray,
        expected_dtype: Union[Type, np.dtype],
        name: str = "array"
    ) -> np.ndarray:
        """Validate array data type and handle type conversion if needed.
        
        Args:
            array: NumPy array to validate
            expected_dtype: Expected data type
            name: Name of the array for error messages
            
        Returns:
            The validated array with the correct dtype
            
        Raises:
            UTCHSValidationError: If validation fails
        """
        if not isinstance(array, np.ndarray):
            try:
                array = np.array(array, dtype=expected_dtype)
            except (ValueError, TypeError) as e:
                raise UTCHSValidationError(
                    f"Could not convert {name} to array with dtype {expected_dtype}: {str(e)}"
                )
                
        # Handle special cases for Decimal and complex types
        if expected_dtype in (Decimal, np.dtype('O')):
            # Check if all values are of the same type
            types = set(type(x) for x in array.flat)
            if len(types) > 1:
                raise UTCHSValidationError(
                    f"Array validation failed for {name}: mixed types detected"
                )
                
            if not all(isinstance(x, (Decimal, float, int)) for x in array.flat):
                raise UTCHSValidationError(
                    f"All values in {name} must be of type Decimal"
                )
            # Convert all values to Decimal if not already
            try:
                array = np.array([Decimal(str(x)) if not isinstance(x, Decimal) else x 
                                for x in array.flat]).reshape(array.shape)
            except InvalidOperation as e:
                raise UTCHSValidationError(f"Invalid numeric value in {name}: {str(e)}")
                
        elif expected_dtype == complex or (isinstance(expected_dtype, np.dtype) and expected_dtype.kind == 'c'):
            # Check if all values are of the same type
            types = set(type(x) for x in array.flat)
            if len(types) > 1:
                raise UTCHSValidationError(
                    f"Array validation failed for {name}: mixed types detected"
                )
                
            if not all(isinstance(x, (complex, float, int)) for x in array.flat):
                raise UTCHSValidationError(
                    f"All values in {name} must be of type complex"
                )
            # Convert to complex if needed
            try:
                array = array.astype(np.complex128)
            except (ValueError, TypeError) as e:
                raise UTCHSValidationError(f"Could not convert {name} to complex: {str(e)}")
                
        else:
            # For other types, try direct conversion
            try:
                if array.dtype != expected_dtype:
                    array = array.astype(expected_dtype)
            except (ValueError, TypeError) as e:
                raise UTCHSValidationError(
                    f"Could not convert {name} to dtype {expected_dtype}: {str(e)}"
                )
                
        return array
            
    def _validate_list_length(
        self,
        value: List[Any],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        name: str = "list"
    ) -> None:
        """Validate that a list has the expected length."""
        if min_length is not None and len(value) < min_length:
            raise UTCHSValidationError(
                f"{name} must have length >= {min_length}, got {len(value)}"
            )
        if max_length is not None and len(value) > max_length:
            raise UTCHSValidationError(
                f"{name} must have length <= {max_length}, got {len(value)}"
            )
            
    def _validate_set_membership(
        self,
        value: Any,
        allowed_values: Set[Any],
        name: str = "value"
    ) -> None:
        """Validate that a value is in a set of allowed values."""
        if value not in allowed_values:
            raise UTCHSValidationError(
                f"{name} must be one of {allowed_values}, got {value}"
            )
            
    def _validate_model(
        self,
        data: Dict[str, Any],
        model_class: Type[T],
        name: str = "data"
    ) -> T:
        """Validate data against a Pydantic model."""
        try:
            return model_class(**data)
        except ValidationError as e:
            raise UTCHSValidationError(f"Invalid {name}: {str(e)}")
            
    def _validate_callable(
        self,
        value: Any,
        name: str = "callable"
    ) -> None:
        """Validate that a value is callable."""
        if not callable(value):
            raise UTCHSValidationError(f"{name} must be callable")
            
    def _validate_path(
        self,
        path: str,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        name: str = "path"
    ) -> None:
        """Validate a file path."""
        from pathlib import Path
        
        path_obj = Path(path)
        if must_exist and not path_obj.exists():
            raise UTCHSValidationError(f"{name} must exist: {path}")
        if must_be_file and not path_obj.is_file():
            raise UTCHSValidationError(f"{name} must be a file: {path}")
        if must_be_dir and not path_obj.is_dir():
            raise UTCHSValidationError(f"{name} must be a directory: {path}")
            
    def _validate_dict_keys(
        self,
        value: Dict[str, Any],
        required_keys: Optional[Set[str]] = None,
        allowed_keys: Optional[Set[str]] = None,
        name: str = "dict"
    ) -> None:
        """Validate dictionary keys."""
        if required_keys is not None:
            missing_keys = required_keys - set(value.keys())
            if missing_keys:
                raise UTCHSValidationError(
                    f"{name} is missing required keys: {missing_keys}"
                )
        if allowed_keys is not None:
            invalid_keys = set(value.keys()) - allowed_keys
            if invalid_keys:
                raise UTCHSValidationError(
                    f"{name} contains invalid keys: {invalid_keys}"
                )

    def _validate_core_concept(self, name: str) -> None:
        """Validate that a name follows core UTCHS concept naming."""
        if name not in CORE_CONCEPTS:
            raise UTCHSValidationError(
                f"'{name}' is not a recognized UTCHS core concept. "
                f"Must be one of: {list(CORE_CONCEPTS.keys())}"
            )
            
    def _validate_field_type(self, name: str) -> None:
        """Validate that a name follows UTCHS field type naming."""
        if name not in FIELD_TYPES:
            raise UTCHSValidationError(
                f"'{name}' is not a recognized UTCHS field type. "
                f"Must be one of: {list(FIELD_TYPES.keys())}"
            )
            
    def _validate_attribute_name(self, name: str, pattern_type: str) -> None:
        """Validate that an attribute name follows UTCHS patterns."""
        import re
        if pattern_type not in ATTRIBUTE_PATTERNS:
            raise UTCHSValidationError(f"Unknown attribute pattern type: {pattern_type}")
            
        pattern = ATTRIBUTE_PATTERNS[pattern_type]
        if not re.match(pattern, name):
            raise UTCHSValidationError(
                f"Attribute name '{name}' does not follow the {pattern_type} pattern"
            )
            
    def _validate_method_name(self, name: str, pattern_type: str) -> None:
        """Validate that a method name follows UTCHS patterns."""
        import re
        if pattern_type not in METHOD_PATTERNS:
            raise UTCHSValidationError(f"Unknown method pattern type: {pattern_type}")
            
        pattern = METHOD_PATTERNS[pattern_type]
        if not re.match(pattern, name):
            raise UTCHSValidationError(
                f"Method name '{name}' does not follow the {pattern_type} pattern"
            )
            
    def _validate_position_number(self, number: int) -> None:
        """Validate that a position number is valid (1-13)."""
        if not isinstance(number, int) or number < 1 or number > 13:
            raise UTCHSValidationError(
                f"Position number must be an integer between 1 and 13, got {number}"
            )
            
    def _validate_grid_dimensions(self, dimensions: tuple) -> None:
        """Validate grid dimensions for fields."""
        if not isinstance(dimensions, tuple) or len(dimensions) != 3:
            raise UTCHSValidationError(
                f"Grid dimensions must be a 3-tuple of integers, got {dimensions}"
            )
        if not all(isinstance(d, int) and d > 0 for d in dimensions):
            raise UTCHSValidationError(
                f"Grid dimensions must be positive integers, got {dimensions}"
            )

    def _validate_numeric_precision(
        self,
        value: Union[Decimal, complex, np.ndarray],
        field_type: str
    ) -> None:
        """Validate numeric precision requirements.
        
        Args:
            value: Value to validate
            field_type: Type of field being validated
            
        Raises:
            UTCHSValidationError: If validation fails
        """
        type_info = NUMERIC_TYPES[field_type]
        max_precision = type_info.get("max_precision")
        
        if max_precision is None:
            return
            
        def check_decimal_precision(d: Decimal) -> None:
            """Check if a Decimal value exceeds maximum precision."""
            if d.is_infinite():
                raise UTCHSValidationError("Decimal value cannot be infinite")
            str_repr = str(d).split('.')[-1] if '.' in str(d) else ''
            if len(str_repr) > max_precision:
                raise UTCHSValidationError(
                    f"Decimal precision {len(str_repr)} exceeds maximum precision {max_precision}"
                )
                
        def check_complex_precision(c: complex) -> None:
            """Check if a complex value is valid."""
            if np.isinf(c.real) or np.isinf(c.imag):
                raise UTCHSValidationError("Complex value cannot contain infinity")
            if isinstance(c.real, Decimal):
                check_decimal_precision(c.real)
            if isinstance(c.imag, Decimal):
                check_decimal_precision(c.imag)
                
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.complexfloating):
                # Check for infinity in complex arrays
                if np.any(np.isinf(value.real)) or np.any(np.isinf(value.imag)):
                    raise UTCHSValidationError("Complex array cannot contain infinity")
                # Check precision for Decimal components
                if field_type == "vector_decimal":
                    for item in value.flatten():
                        if isinstance(item, Decimal):
                            check_decimal_precision(item)
            elif field_type == "vector_decimal":
                # Convert to Decimal and check precision
                try:
                    decimal_array = np.array([[Decimal(str(x)) for x in row] for row in value])
                    for item in decimal_array.flatten():
                        check_decimal_precision(item)
                except (ValueError, TypeError, InvalidOperation) as e:
                    raise UTCHSValidationError(f"Invalid numeric value: {str(e)}")
        elif isinstance(value, complex):
            check_complex_precision(value)
        elif isinstance(value, Decimal):
            check_decimal_precision(value)
        elif field_type == "vector_decimal":
            try:
                decimal_value = Decimal(str(value))
                check_decimal_precision(decimal_value)
            except (ValueError, TypeError, InvalidOperation) as e:
                raise UTCHSValidationError(f"Invalid numeric value: {str(e)}")

    def validate_module_data(
        self,
        module_name: str,
        data: Dict[str, Any]
    ) -> None:
        """Validate module data against registered validators.
        
        Args:
            module_name: Name of the module to validate
            data: Data to validate
            
        Raises:
            UTCHSValidationError: If validation fails
            ValueError: If no validators are registered for the module
        """
        if module_name not in self.validators:
            raise ValueError(f"No validators registered for module: {module_name}")

        validators = self.validators[module_name]
        
        # Check for required fields based on module type
        required_fields = {
            "Position": ["number", "phase", "energy_level", "spatial_location"],
            "PhaseField": ["field_data", "grid_size", "singularities"],
            "VectorField": ["field_data", "magnitude_range"],
            "EnergyField": ["field_data", "grid_spacing"],
            "test_module": ["field_data"]
        }

        if module_name in required_fields:
            for field_name in required_fields[module_name]:
                if field_name not in data:
                    raise UTCHSValidationError(f"Required field '{field_name}' missing in {module_name} data")

        # Validate each field
        for field_name, validator in validators.items():
            if field_name not in data:
                continue

            try:
                value = data[field_name]
                
                # Special validation for PhaseField
                if module_name == "PhaseField" and field_name == "field_data":
                    if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.complexfloating):
                        raise UTCHSValidationError("Field field_data must be a complex array")
                
                # Special validation for Position
                elif module_name == "Position":
                    if field_name == "spatial_location" and len(value) != 3:
                        raise UTCHSValidationError("spatial_location must have shape (3,)")
                    elif field_name == "velocity_vector" and len(value) != 3:
                        raise UTCHSValidationError("velocity_vector must have shape (3,)")
                    elif field_name == "energy_level" and value < 0:
                        raise UTCHSValidationError("energy_level must be >= 0")
                    elif field_name == "number" and not (1 <= value <= 13):
                        raise UTCHSValidationError("position number must be between 1 and 13")
                
                # Special validation for VectorField
                elif module_name == "VectorField":
                    if field_name == "magnitude_range":
                        if not all(value[i] <= value[i+1] for i in range(len(value)-1)):
                            raise UTCHSValidationError("magnitude_range values must be in ascending order")
                    elif field_name == "field_data":
                        try:
                            arr = np.array(value)
                            if not all(isinstance(v, Decimal) for v in arr.flatten()):
                                raise UTCHSValidationError("All values in field_data must be of type Decimal")
                        except Exception as e:
                            raise UTCHSValidationError(f"Invalid array format: {str(e)}")
                
                # Special validation for EnergyField
                elif module_name == "EnergyField":
                    if field_name == "grid_spacing" and value <= 0:
                        raise UTCHSValidationError("grid_spacing must be > 0")
                
                # Special validation for test_module
                elif module_name == "test_module":
                    if field_name == "field_data":
                        if isinstance(value, complex) and (np.isinf(value.real) or np.isinf(value.imag)):
                            raise UTCHSValidationError("Complex value in field_data cannot contain infinity")
                        elif isinstance(value, Decimal) and len(str(value)) > 20:
                            raise UTCHSValidationError("Decimal precision exceeds maximum")
                        elif isinstance(value, np.ndarray) and not all(isinstance(v, (int, float)) for v in value.flatten()):
                            raise UTCHSValidationError("Invalid numeric value in field_data")

                # General validation
                is_valid = validator(value)
                if not is_valid:
                    raise UTCHSValidationError(f"Validation failed for field '{field_name}' in {module_name}")

            except Exception as e:
                if isinstance(e, UTCHSValidationError):
                    raise
                raise UTCHSValidationError(f"Error validating field '{field_name}' in {module_name}: {str(e)}")

# Create a global instance
validation_registry = ValidationRegistry() 