"""
Möbius transformation module for the UTCHS framework.

This module implements the MobiusTransformation class, which forms the core
mathematical structure for phase recursion within the UTCHS framework.
"""

import cmath
import numpy as np
from typing import List, Optional, Tuple, Union

class MobiusTransformation:
    """
    Implements Möbius transformations for phase evolution.

    A Möbius transformation has the form f(z) = (az + b)/(cz + d),
    where a, b, c, d are complex numbers with ad - bc ≠ 0.
    """
    def __init__(
        self,
        a: complex = complex(1, 0),
        b: complex = complex(0, 0),
        c: complex = complex(0, 0),
        d: complex = complex(1, 0)
    ):
        """
        Initialize with Möbius transformation parameters.

        Args:
            a, b, c, d: Complex parameters for the transformation
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        # Validate that ad - bc ≠ 0
        determinant = a * d - b * c
        if abs(determinant) < 1e-10:
            raise ValueError("Invalid Möbius transformation: ad - bc must be non-zero")

    def transform(self, z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
        """
        Apply the Möbius transformation to a complex value or array.

        Args:
            z: Complex value or array to transform

        Returns:
            Transformed complex value or array
        """
        # Handle both scalar and array inputs
        if isinstance(z, np.ndarray):
            # For array inputs, we need to handle potential division by zero
            denominator = self.c * z + self.d
            # Create a mask where denominator is close to zero
            mask = np.abs(denominator) < 1e-10
            
            # Initialize result array
            result = np.zeros_like(z)
            
            # Process non-zero denominators
            non_zero_mask = ~mask
            if np.any(non_zero_mask):
                result[non_zero_mask] = (self.a * z[non_zero_mask] + self.b) / denominator[non_zero_mask]
                
            # Handle zero denominators (map to infinity, represented as large value)
            if np.any(mask):
                # Direction of infinity depends on numerator
                numerator = self.a * z[mask] + self.b
                dir_mask = np.abs(numerator) < 1e-10
                result[mask & dir_mask] = complex(0, 0)  # 0/0 form, undefined
                result[mask & ~dir_mask] = numerator[~dir_mask] * 1e10  # Non-zero/0 form, infinity in direction of numerator
                
            return result
        else:
            # For scalar input
            denominator = self.c * z + self.d
            if abs(denominator) < 1e-10:
                # Handle division by zero
                numerator = self.a * z + self.b
                if abs(numerator) < 1e-10:
                    return complex(0, 0)  # Undefined case (0/0)
                else:
                    # Return "infinity" in the direction of numerator
                    return numerator * 1e10
            return (self.a * z + self.b) / denominator

    def compose(self, other: 'MobiusTransformation') -> 'MobiusTransformation':
        """
        Compose this transformation with another.

        Args:
            other: Another MobiusTransformation

        Returns:
            A new MobiusTransformation representing the composition
        """
        a = self.a * other.a + self.b * other.c
        b = self.a * other.b + self.b * other.d
        c = self.c * other.a + self.d * other.c
        d = self.c * other.b + self.d * other.d

        return MobiusTransformation(a, b, c, d)

    def inverse(self) -> 'MobiusTransformation':
        """
        Return the inverse transformation.

        Returns:
            The inverse MobiusTransformation
        """
        return MobiusTransformation(self.d, -self.b, -self.c, self.a)

    def fixed_points(self) -> List[complex]:
        """
        Calculate the fixed points of this transformation.

        Returns:
            List of fixed points (0, 1, or 2 points)
        """
        if abs(self.c) < 1e-10:
            # c = 0: one fixed point or identity transformation
            if abs(self.a - self.d) < 1e-10:
                # Identity transformation: all points are fixed (return empty list as special case)
                return []
            else:
                # One fixed point: b/(d-a)
                return [self.b / (self.d - self.a)]
        else:
            # Solve the quadratic equation: cz^2 + (d-a)z - b = 0
            discriminant = (self.d - self.a)**2 + 4 * self.b * self.c
            if abs(discriminant) < 1e-10:
                # One repeated fixed point
                return [(self.a - self.d) / (2 * self.c)]
            else:
                # Two distinct fixed points
                sqrt_discriminant = cmath.sqrt(discriminant)
                z1 = ((self.a - self.d) + sqrt_discriminant) / (2 * self.c)
                z2 = ((self.a - self.d) - sqrt_discriminant) / (2 * self.c)
                return [z1, z2]

    def iterate(self, z0: complex, n: int) -> List[complex]:
        """
        Iterate the transformation starting from z0 for n steps.

        Args:
            z0: Initial complex value
            n: Number of iterations

        Returns:
            List of iterated values [z0, z1, z2, ..., zn]
        """
        result = [z0]
        z = z0
        for _ in range(n):
            z = self.transform(z)
            result.append(z)
        return result

    def get_params_dict(self) -> dict:
        """
        Return the transformation parameters as a dictionary.

        Returns:
            Dictionary with a, b, c, d parameters
        """
        return {
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'd': self.d
        }

    @classmethod
    def from_fixed_points(cls, z1: complex, z2: complex, w1: complex, w2: complex) -> 'MobiusTransformation':
        """
        Create a Möbius transformation that maps z1 to w1 and z2 to w2.

        Args:
            z1, z2: Source points
            w1, w2: Target points

        Returns:
            MobiusTransformation mapping z1→w1 and z2→w2
        """
        # Edge case: z1 = z2 or w1 = w2 is not allowed
        if abs(z1 - z2) < 1e-10 or abs(w1 - w2) < 1e-10:
            raise ValueError("Source and target points must be distinct")
            
        # Calculate parameters
        a = (w1 - w2) * z1
        b = -(w1 - w2) * z1 * z2
        c = w1 - w2
        d = -w1 * z2 + w2 * z1
        
        return cls(a, b, c, d)
