"""
Möbius transformation module for the UTCHS framework.

This module implements the MobiusTransformation class, which applies Möbius
transformations to phase fields, including recursive logic structures based
on the OmniLens Möbius Framework.
"""

import cmath
import math
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Any

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class MobiusNode:
    """
    A node representing a single point in a Möbius recursion network.
    
    Implements recursive phase logic for individual field points based on the
    OmniLens Möbius Framework concept.
    
    Attributes:
        signal: The data, intent, or symbolic input
        phase: Recursive harmonic phase state
        next: Möbius wrap link for connecting nodes
        feedback: Recursive self-reflection link
    """
    def __init__(self, signal: complex, phase: float = 0):
        self.signal = signal      # signal = data, intent, or symbolic input
        self.phase = phase        # recursive harmonic phase state
        self.next = None          # Möbius wrap link
        self.feedback = None      # recursive self-reflection link

    def recurse(self, field_harmonic: float) -> float:
        """
        Recursively update the node's phase based on field interactions.
        
        Args:
            field_harmonic: The harmonic alignment value from the field
            
        Returns:
            Coherence value resulting from the recursion
        """
        # Phase-lock to field resonance
        coherence = abs(self.signal) * field_harmonic
        self.phase = (self.phase + coherence) % (2 * np.pi)  # wrap phase cycle
        return coherence

    def apply_mobius(self, a: complex, b: complex, c: complex, d: complex) -> None:
        """
        Apply a Möbius transformation to the node's signal.
        
        Args:
            a, b, c, d: Möbius transformation parameters
        """
        if abs(c * self.signal + d) < 1e-10:
            # Avoid division by zero
            self.signal = float('inf')
        else:
            self.signal = (a * self.signal + b) / (c * self.signal + d)

class MobiusTransformation:
    """
    Implements Möbius transformations on complex-valued fields.
    
    A Möbius transformation is a function of the form:
    f(z) = (az + b) / (cz + d)
    where a, b, c, d are complex numbers with ad - bc ≠ 0.
    
    This implementation includes recursive node-based processing capability and
    correction terms for higher recursion orders.
    """
    
    def __init__(self, a: complex, b: complex, c: complex, d: complex, 
                recursion_order: int = 1, 
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize a Möbius transformation with parameters a, b, c, d.
        
        Args:
            a, b, c, d: Complex parameters defining the transformation
            recursion_order: Order of recursion (1=base level, 2=meta, etc.)
            config: Configuration dictionary (optional)
            
        Raises:
            ValueError: If ad - bc = 0 (degenerate transformation)
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.recursion_order = recursion_order
        self.config = config or {}
        
        # Check that the transformation is non-degenerate
        det = a * d - b * c
        if abs(det) < 1e-10:
            raise ValueError("Degenerate Möbius transformation: ad - bc ≈ 0")
        
        # Store determinant for later use
        self.determinant = det
        
        # Initialize node-based recursion attributes
        self.nodes = {}
        self.field_harmonic_function = lambda phase: 1 - abs((phase % (2 * np.pi)) - np.pi) / np.pi
        
        # Numeric stability parameters
        self.min_denominator = self.config.get('min_denominator', 1e-10)
        
        # Enable correction terms for higher recursion orders
        self.enable_correction = self.config.get('enable_correction', True)
        
        # Golden ratio (φ) for correction calculations
        self.phi = (1 + math.sqrt(5)) / 2
            
    def transform(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the Möbius transformation to a complex array.
        
        Args:
            z: Complex-valued array
            
        Returns:
            Transformed complex-valued array
        """
        # Apply correction terms if enabled for higher recursion orders
        a, b, c, d = self.get_corrected_parameters()
        
        # Handle division by zero by replacing with infinity
        denominator = c * z + d
        result = np.zeros_like(z, dtype=complex)
        
        # Handle points where denominator is close to zero
        near_zero = np.abs(denominator) < self.min_denominator
        result[near_zero] = np.inf
        
        # Transform other points normally
        non_zero = ~near_zero
        result[non_zero] = (a * z[non_zero] + b) / denominator[non_zero]
        
        return result
    
    def get_corrected_parameters(self) -> Tuple[complex, complex, complex, complex]:
        """
        Get Möbius parameters with correction terms applied.
        
        Returns:
            Corrected parameters (a, b, c, d)
        """
        # No correction for base level or if disabled
        if self.recursion_order <= 1 or not self.enable_correction:
            return self.a, self.b, self.c, self.d
        
        # Calculate correction factor based on recursion order
        correction_factor = self._calculate_correction_factor()
        
        # Apply correction to parameters (preserving determinant)
        # We apply corrections that maintain the determinant a*d - b*c
        a_corrected = self.a * (1 + correction_factor * 0.1)
        d_corrected = self.d * (1 - correction_factor * 0.1)
        
        # For b and c, apply minimal corrections to maintain stability
        # These corrections are designed to preserve key fixed points
        b_corrected = self.b * (1 - correction_factor * 0.05)
        c_corrected = self.c * (1 + correction_factor * 0.05)
        
        # Ensure determinant is preserved
        new_det = a_corrected * d_corrected - b_corrected * c_corrected
        det_ratio = self.determinant / new_det if abs(new_det) > 1e-10 else 1.0
        
        # Apply determinant correction
        a_corrected *= det_ratio**0.25
        d_corrected *= det_ratio**0.25
        b_corrected *= det_ratio**0.25
        c_corrected *= det_ratio**0.25
        
        return a_corrected, b_corrected, c_corrected, d_corrected
    
    def _calculate_correction_factor(self) -> float:
        """
        Calculate the correction factor based on recursion order.
        
        Returns:
            Correction factor (0.0-1.0)
        """
        # Base correction increases with recursion order but plateaus
        base_correction = 1.0 - 1.0 / self.recursion_order
        
        # Apply logarithmic scaling to avoid excessive correction at high orders
        log_factor = math.log(self.recursion_order + 1) / math.log(10)
        log_correction = min(base_correction * log_factor, 0.5)
        
        # Phi-based resonance: corrections are stronger at fibonacci-related recursion orders
        phi_power = self.recursion_order - 1
        fib_approx = self.phi ** phi_power
        closest_fib = round(fib_approx)
        phi_resonance = 1.0 - min(abs(fib_approx - closest_fib) / fib_approx, 0.5)
        
        # Combine corrections with phi resonance having more weight
        combined_correction = 0.4 * base_correction + 0.2 * log_correction + 0.4 * phi_resonance
        
        # Limit maximum correction
        return min(combined_correction, 0.5)

    def transform_single(self, z: complex) -> complex:
        """
        Apply the Möbius transformation to a single complex value.
        
        Args:
            z: Complex value
            
        Returns:
            Transformed complex value
        """
        a, b, c, d = self.get_corrected_parameters()
        
        if abs(c * z + d) < self.min_denominator:
            return float('inf')
            
        return (a * z + b) / (c * z + d)
        
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
        
        # Use maximum recursion order for composed transform
        max_order = max(self.recursion_order, other.recursion_order)

        return MobiusTransformation(a, b, c, d, recursion_order=max_order, config=self.config)

    def inverse(self) -> 'MobiusTransformation':
        """
        Return the inverse transformation.

        Returns:
            The inverse MobiusTransformation
        """
        return MobiusTransformation(
            self.d, -self.b, -self.c, self.a, 
            recursion_order=self.recursion_order,
            config=self.config
        )

    def fixed_points(self) -> List[complex]:
        """
        Calculate the fixed points of this transformation.

        Returns:
            List of fixed points (0, 1, or 2 points)
        """
        # Get corrected parameters for accurate fixed point calculation
        a, b, c, d = self.get_corrected_parameters()
        
        if abs(c) < self.min_denominator:
            # c = 0: one fixed point or identity transformation
            if abs(a - d) < self.min_denominator:
                # Identity transformation: all points are fixed (return empty list as special case)
                return []
            else:
                # One fixed point: b/(d-a)
                return [b / (d - a)]
        else:
            # Solve the quadratic equation: cz^2 + (d-a)z - b = 0
            discriminant = (d - a)**2 + 4 * b * c
            if abs(discriminant) < self.min_denominator:
                # One repeated fixed point
                return [(a - d) / (2 * c)]
            else:
                # Two distinct fixed points
                sqrt_discriminant = cmath.sqrt(discriminant)
                z1 = ((a - d) + sqrt_discriminant) / (2 * c)
                z2 = ((a - d) - sqrt_discriminant) / (2 * c)
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
            z = self.transform_single(z)
            if z == float('inf'):
                break
            result.append(z)
        return result

    def get_params_dict(self) -> dict:
        """
        Return the transformation parameters as a dictionary.

        Returns:
            Dictionary with a, b, c, d parameters and recursion order
        """
        return {
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'd': self.d,
            'recursion_order': self.recursion_order,
            'determinant': self.determinant
        }

    @classmethod
    def from_fixed_points(cls, z1: complex, z2: complex, w1: complex, w2: complex, 
                         recursion_order: int = 1,
                         config: Optional[Dict[str, Any]] = None) -> 'MobiusTransformation':
        """
        Create a Möbius transformation that maps z1 to w1 and z2 to w2.

        Args:
            z1, z2: Source points
            w1, w2: Target points
            recursion_order: Order of recursion (1=base level, 2=meta, etc.)
            config: Configuration dictionary (optional)

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
        
        return cls(a, b, c, d, recursion_order=recursion_order, config=config)
