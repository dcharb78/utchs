"""
Position module for the UTCHS framework.

This module implements the Position class, which represents a fundamental unit
in the UTCHS framework. Each position (1-13) has specific properties and roles.
"""

import numpy as np
from typing import Dict, List, Optional

class Position:
    """
    Represents a fundamental unit in the UTCHS framework.

    Each position (1-13) has specific properties and roles within the system.
    """
    def __init__(
        self,
        number: int,
        role: str,
        spatial_location: np.ndarray,
        rotational_angle: float,
        energy_level: float,
        absolute_position: Dict[str, int],
        relations: Dict[str, List[int]]
    ):
        """
        Initialize a position with its multidimensional properties.

        Args:
            number: Position number (1-13)
            role: Role description (e.g., "Origin Point", "Seed Position")
            spatial_location: 3D coordinates [x, y, z]
            rotational_angle: Position's angle within cycle
            energy_level: Current energy value
            absolute_position: Dict with torus, structure, cycle, position indices
            relations: Dict of relationships to other positions
        """
        self.number = number
        self.role = role
        self.spatial_location = spatial_location
        self.rotational_angle = rotational_angle
        self.energy_level = energy_level
        self.absolute_position = absolute_position
        self.relations = relations
        self.digital_root = self._calculate_digital_root()
        self.phase = 0.0  # Current phase value

    def _calculate_digital_root(self) -> int:
        """Calculate the digital root of the position number."""
        if self.number <= 9:
            return self.number
        elif self.number == 10:
            return 1
        elif self.number == 11:
            return 2
        elif self.number == 12:
            return 3
        elif self.number == 13:
            return 4

    def update_phase(self, mobius_parameters: Dict[str, complex]) -> None:
        """
        Update the phase using a Möbius transformation.

        Args:
            mobius_parameters: Parameters a, b, c, d for the Möbius transformation
        """
        a, b, c, d = mobius_parameters['a'], mobius_parameters['b'], mobius_parameters['c'], mobius_parameters['d']
        z = complex(self.phase, 0)
        new_z = (a * z + b) / (c * z + d)
        self.phase = float(new_z.real)

    def get_harmonic_signature(self) -> np.ndarray:
        """
        Return the harmonic signature of this position.
        
        The harmonic signature is a vector of values representing the resonance
        characteristics of this position based on its number, digital root,
        and phase relationships.
        
        Returns:
            NumPy array of signature values
        """
        # Initialize harmonic signature array
        signature = np.zeros(13)
        
        # Position number influences primary resonance
        signature[self.number - 1] = 1.0
        
        # Digital root influences secondary resonances
        digital_root_position = self.digital_root - 1
        if digital_root_position >= 0 and digital_root_position < 13:
            signature[digital_root_position] += 0.5
            
        # Phase value influences overall signature
        phase_factor = 0.3 * (np.sin(self.phase) + 1) / 2
        signature = signature * (1.0 + phase_factor)
        
        # Special amplification for vortex positions (3, 6, 9)
        if self.number in [3, 6, 9]:
            signature = signature * 1.5
            
        # Normalize signature
        signature = signature / np.sum(signature)
        
        return signature

    def get_vortex_category(self) -> str:
        """
        Determine the vortex category of this position.
        
        Returns:
            String describing the vortex category
        """
        if self.number in [3, 6, 9]:
            return "Vortex Point"
        elif self.number in [1, 2, 4, 5, 7, 8, 10, 11, 12, 13]:
            return "Regular Point"
        else:
            return "Unknown"

    def calculate_resonance_with(self, other_position: 'Position') -> float:
        """
        Calculate the resonance between this position and another.
        
        Args:
            other_position: Another position to calculate resonance with
            
        Returns:
            Resonance value between 0 and 1
        """
        # Get harmonic signatures
        signature1 = self.get_harmonic_signature()
        signature2 = other_position.get_harmonic_signature()
        
        # Calculate cosine similarity
        dot_product = np.sum(signature1 * signature2)
        norm1 = np.sqrt(np.sum(signature1**2))
        norm2 = np.sqrt(np.sum(signature2**2))
        
        if norm1 > 0 and norm2 > 0:
            resonance = dot_product / (norm1 * norm2)
        else:
            resonance = 0.0
            
        # Apply position-specific modifiers
        if self.number == other_position.number:
            resonance *= 1.2  # Boost resonance for same position number
            
        if self.digital_root == other_position.digital_root:
            resonance *= 1.1  # Boost resonance for same digital root
            
        # Normalize to [0, 1]
        resonance = np.clip(resonance, 0.0, 1.0)
        
        return resonance
