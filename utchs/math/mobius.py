"""
Möbius transformation module for the UTCHS framework.

This module implements the MobiusTransformation class, which applies Möbius
transformations to phase fields, including recursive logic structures based
on the OmniLens Möbius Framework.
"""

import cmath
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Any

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
    
    This implementation includes recursive node-based processing capability.
    """
    
    def __init__(self, a: complex, b: complex, c: complex, d: complex):
        """
        Initialize a Möbius transformation with parameters a, b, c, d.
        
        Args:
            a, b, c, d: Complex parameters defining the transformation
            
        Raises:
            ValueError: If ad - bc = 0 (degenerate transformation)
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        # Check that the transformation is non-degenerate
        det = a * d - b * c
        if abs(det) < 1e-10:
            raise ValueError("Degenerate Möbius transformation: ad - bc ≈ 0")
        
        # Initialize node-based recursion attributes
        self.nodes = {}
        self.field_harmonic_function = lambda phase: 1 - abs((phase % (2 * np.pi)) - np.pi) / np.pi
            
    def transform(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the Möbius transformation to a complex array.
        
        Args:
            z: Complex-valued array
            
        Returns:
            Transformed complex-valued array
        """
        # Handle division by zero by replacing with infinity
        denominator = self.c * z + self.d
        result = np.zeros_like(z, dtype=complex)
        
        # Handle points where denominator is close to zero
        near_zero = np.abs(denominator) < 1e-10
        result[near_zero] = np.inf
        
        # Transform other points normally
        non_zero = ~near_zero
        result[non_zero] = (self.a * z[non_zero] + self.b) / denominator[non_zero]
        
        return result
    
    def transform_recursive(self, z: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Apply the Möbius transformation with recursive node-based processing.
        
        Args:
            z: Complex-valued array
            iterations: Number of recursive iterations to perform
            
        Returns:
            Transformed complex-valued array with recursive processing
        """
        # Initialize nodes if not already created
        shape = z.shape
        flat_z = z.flatten()
        
        # Create or update nodes
        for i, val in enumerate(flat_z):
            if i not in self.nodes:
                self.nodes[i] = MobiusNode(val)
            else:
                self.nodes[i].signal = val
                
        # Perform recursive iterations
        for _ in range(iterations):
            for i, node in self.nodes.items():
                # Apply Möbius transformation to node
                node.apply_mobius(self.a, self.b, self.c, self.d)
                
                # Perform recursive phase update
                harmonic = self.field_harmonic_function(node.phase)
                node.recurse(harmonic)
                
                # Update the result
                flat_z[i] = node.signal
        
        # Reshape and return the result
        return flat_z.reshape(shape)
    
    @staticmethod
    def from_fixed_points(z1: complex, z2: complex, z3: complex,
                          w1: complex, w2: complex, w3: complex) -> 'MobiusTransformation':
        """
        Create a Möbius transformation that maps three points to three other points.
        
        Args:
            z1, z2, z3: Source points
            w1, w2, w3: Target points
            
        Returns:
            MobiusTransformation that maps z1→w1, z2→w2, z3→w3
            
        Raises:
            ValueError: If any points are coincident or the mapping is degenerate
        """
        # Check for coincident points
        if z1 == z2 or z1 == z3 or z2 == z3:
            raise ValueError("Source points must be distinct")
        if w1 == w2 or w1 == w3 or w2 == w3:
            raise ValueError("Target points must be distinct")
        
        # Calculate cross-ratios to ensure a valid mapping
        z_cr = (z1 - z3) * (z2 - z3) / ((z1 - z2) * (z3 - z3))
        w_cr = (w1 - w3) * (w2 - w3) / ((w1 - w2) * (w3 - w3))
        
        if abs(z_cr - w_cr) > 1e-10:
            raise ValueError("No Möbius transformation can map these triples")
        
        # Calculate transformation parameters
        a = w1 * (z2 - z3) - w2 * (z1 - z3) + w3 * (z1 - z2)
        b = -w1 * z2 * z3 + w2 * z1 * z3 - w3 * z1 * z2
        c = z2 - z3 - z1 + z3 + z1 - z2
        d = -z2 * z3 + z1 * z3 - z1 * z2
        
        return MobiusTransformation(a, b, c, d)

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
