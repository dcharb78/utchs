"""Validation utilities for the UTCHS framework."""

from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

import numpy as np
from pydantic import BaseModel, ValidationError

from utchs.core.exceptions import ValidationError as UTCHSValidationError

T = TypeVar("T", bound=BaseModel)


def validate_type(value: Any, expected_type: Type[Any], name: str = "value") -> None:
    """Validate that a value is of the expected type.

    Args:
        value: Value to validate
        expected_type: Expected type
        name: Name of the value for error messages

    Raises:
        UTCHSValidationError: If validation fails
    """
    if not isinstance(value, expected_type):
        raise UTCHSValidationError(
            f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}"
        )


def validate_range(
    value: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    name: str = "value",
) -> None:
    """Validate that a numeric value is within a range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Name of the value for error messages

    Raises:
        UTCHSValidationError: If validation fails
    """
    if min_value is not None and value < min_value:
        raise UTCHSValidationError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise UTCHSValidationError(f"{name} must be <= {max_value}, got {value}")


def validate_array_shape(
    array: np.ndarray,
    expected_shape: tuple,
    name: str = "array",
) -> None:
    """Validate that a numpy array has the expected shape.

    Args:
        array: Array to validate
        expected_shape: Expected shape
        name: Name of the array for error messages

    Raises:
        UTCHSValidationError: If validation fails
    """
    if array.shape != expected_shape:
        raise UTCHSValidationError(
            f"{name} must have shape {expected_shape}, got {array.shape}"
        )


def validate_array_dtype(
    array: np.ndarray,
    expected_dtype: Union[Type, np.dtype],
    name: str = "array",
) -> None:
    """Validate that a numpy array has the expected dtype.

    Args:
        array: Array to validate
        expected_dtype: Expected dtype
        name: Name of the array for error messages

    Raises:
        UTCHSValidationError: If validation fails
    """
    if not np.issubdtype(array.dtype, expected_dtype):
        raise UTCHSValidationError(
            f"{name} must have dtype {expected_dtype}, got {array.dtype}"
        )


def validate_list_length(
    value: List[Any],
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    name: str = "list",
) -> None:
    """Validate that a list has the expected length.

    Args:
        value: List to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        name: Name of the list for error messages

    Raises:
        UTCHSValidationError: If validation fails
    """
    if min_length is not None and len(value) < min_length:
        raise UTCHSValidationError(
            f"{name} must have length >= {min_length}, got {len(value)}"
        )
    if max_length is not None and len(value) > max_length:
        raise UTCHSValidationError(
            f"{name} must have length <= {max_length}, got {len(value)}"
        )


def validate_set_membership(
    value: Any,
    allowed_values: Set[Any],
    name: str = "value",
) -> None:
    """Validate that a value is in a set of allowed values.

    Args:
        value: Value to validate
        allowed_values: Set of allowed values
        name: Name of the value for error messages

    Raises:
        UTCHSValidationError: If validation fails
    """
    if value not in allowed_values:
        raise UTCHSValidationError(
            f"{name} must be one of {allowed_values}, got {value}"
        )


def validate_model(
    data: Dict[str, Any],
    model_class: Type[T],
    name: str = "data",
) -> T:
    """Validate data against a Pydantic model.

    Args:
        data: Data to validate
        model_class: Pydantic model class
        name: Name of the data for error messages

    Returns:
        Validated model instance

    Raises:
        UTCHSValidationError: If validation fails
    """
    try:
        return model_class(**data)
    except ValidationError as e:
        raise UTCHSValidationError(f"Invalid {name}: {str(e)}")


def validate_callable(
    value: Any,
    name: str = "callable",
) -> None:
    """Validate that a value is callable.

    Args:
        value: Value to validate
        name: Name of the value for error messages

    Raises:
        UTCHSValidationError: If validation fails
    """
    if not callable(value):
        raise UTCHSValidationError(f"{name} must be callable")


def validate_path(
    path: str,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    name: str = "path",
) -> None:
    """Validate a file path.

    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        must_be_file: Whether the path must be a file
        must_be_dir: Whether the path must be a directory
        name: Name of the path for error messages

    Raises:
        UTCHSValidationError: If validation fails
    """
    from pathlib import Path

    path_obj = Path(path)
    if must_exist and not path_obj.exists():
        raise UTCHSValidationError(f"{name} must exist: {path}")
    if must_be_file and not path_obj.is_file():
        raise UTCHSValidationError(f"{name} must be a file: {path}")
    if must_be_dir and not path_obj.is_dir():
        raise UTCHSValidationError(f"{name} must be a directory: {path}")


def validate_dict_keys(
    value: Dict[str, Any],
    required_keys: Optional[Set[str]] = None,
    allowed_keys: Optional[Set[str]] = None,
    name: str = "dict",
) -> None:
    """Validate dictionary keys.

    Args:
        value: Dictionary to validate
        required_keys: Set of required keys
        allowed_keys: Set of allowed keys
        name: Name of the dictionary for error messages

    Raises:
        UTCHSValidationError: If validation fails
    """
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