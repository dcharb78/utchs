"""
Cycle module for the UTCHS framework.

This module implements the Cycle class, which manages the progression through
the 13 positions that form a complete cycle within the UTCHS framework.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from utchs.core.position import Position

class Cycle:
    """
    Represents a complete rotation through all 13 positions.
    """
    def __init__(self, id: int, torus_id: int, structure_id: int):
        """
        Initialize a cycle within a scaled structure.

        Args:
            id: Cycle identifier within structure
            torus_id: Parent torus identifier
            structure_id: Parent structure identifier
        """
        self.id = id
        self.torus_id = torus_id
        self.structure_id = structure_id
        self.positions = self._initialize_positions()
        self.current_position_idx = 0
        self.completed = False

    def _initialize_positions(self) -> List[Position]:
        """Create the 13 positions within this cycle."""
        positions = []
        for i in range(1, 14):
            # Create position with appropriate parameters based on cycle, structure, torus
            role = self._get_position_role(i)
            spatial_location = self._calculate_spatial_location(i)
            rotational_angle = (i - 1) * (2 * np.pi / 13)
            energy_level = self._calculate_energy_level(i)
            absolute_position = {
                "torus": self.torus_id,
                "structure": self.structure_id,
                "cycle": self.id,
                "position": i
            }
            relations = self._calculate_relations(i)
            
            position = Position(
                number=i,
                role=role,
                spatial_location=spatial_location,
                rotational_angle=rotational_angle,
                energy_level=energy_level,
                absolute_position=absolute_position,
                relations=relations
            )
            positions.append(position)
        return positions

    def _get_position_role(self, position_number: int) -> str:
        """
        Get the role description for a position.
        
        Args:
            position_number: Position number (1-13)
            
        Returns:
            Role description string
        """
        roles = {
            1: "Prime Unity Source",
            2: "First Polarity Split",
            3: "First Vortex Point",
            4: "Stabilization Point",
            5: "Prime Anchor Point",
            6: "Second Vortex Point",
            7: "Prime Anchor Point",
            8: "Stabilization Point",
            9: "Third Vortex Point",
            10: "Nested Seed Point",
            11: "Prime Anchor Point",
            12: "Completion Preparation",
            13: "Cycle Completion"
        }
        return roles.get(position_number, "Unknown Position")

    def _calculate_spatial_location(self, position_number: int) -> np.ndarray:
        """
        Calculate the spatial location for a position.
        
        Args:
            position_number: Position number (1-13)
            
        Returns:
            3D spatial coordinates [x, y, z]
        """
        # Map positions to a 3D torus
        angle = (position_number - 1) * (2 * np.pi / 13)
        
        # Calculate position based on cycle id (using golden ratio scaling)
        cycle_factor = self.id * 0.618  # Golden ratio factor
        structure_factor = self.structure_id * 0.382  # 1 - golden ratio
        
        # Major and minor radii
        R = 2.0 + 0.5 * cycle_factor
        r = 0.5 + 0.2 * structure_factor
        
        # Calculate 3D coordinates on torus
        x = (R + r * np.cos(angle)) * np.cos(cycle_factor)
        y = (R + r * np.cos(angle)) * np.sin(cycle_factor)
        z = r * np.sin(angle)
        
        return np.array([x, y, z])

    def _calculate_energy_level(self, position_number: int) -> float:
        """
        Calculate the initial energy level for a position.
        
        Args:
            position_number: Position number (1-13)
            
        Returns:
            Energy value
        """
        # Base energy level
        base_energy = 1.0
        
        # Special positions (prime numbers, vortex points) have higher energy
        if position_number in [1, 2, 3, 5, 7, 11, 13]:  # Prime numbers
            base_energy *= 1.5
            
        if position_number in [3, 6, 9]:  # Vortex points
            base_energy *= 1.3
            
        if position_number == 10:  # Seed point
            base_energy *= 1.8
            
        # Cycle and structure influence
        cycle_factor = 1.0 - (0.05 * (self.id - 1))  # Energy decreases with each cycle
        
        # Apply quantum scaling functions
        quantum_factor = 0.9 + 0.2 * np.sin(position_number * np.pi / 6.5)
        
        return base_energy * cycle_factor * quantum_factor

    def _calculate_relations(self, position_number: int) -> Dict[str, List[int]]:
        """
        Calculate relationships to other positions.
        
        Args:
            position_number: Position number (1-13)
            
        Returns:
            Dictionary of relationships
        """
        relations = {
            "harmonic": [],  # Positions with harmonic relationships
            "resonant": [],  # Positions with resonance
            "dual": []       # Positions with duality relationships
        }
        
        # Harmonic relationships based on musical theory
        for other_number in range(1, 14):
            if other_number == position_number:
                continue
                
            # Digital roots
            position_digital_root = (position_number - 1) % 9 + 1
            other_digital_root = (other_number - 1) % 9 + 1
            
            digital_root_sum = position_digital_root + other_digital_root
            
            # Define harmonic relationships
            if digital_root_sum == 10:  # Octave relationship
                relations["harmonic"].append(other_number)
                
            # Resonance based on 3-6-9 pattern
            if position_number in [3, 6, 9] and other_number in [3, 6, 9]:
                relations["resonant"].append(other_number)
                
            # Duality relationships
            if position_number + other_number == 14:  # Positions that sum to 14 are dual
                relations["dual"].append(other_number)
        
        return relations

    def advance(self) -> Tuple[bool, Optional[Position]]:
        """
        Advance to the next position in the cycle.

        Returns:
            Tuple of (cycle_completed, current_position)
        """
        self.current_position_idx = (self.current_position_idx + 1) % 13
        if self.current_position_idx == 0:
            self.completed = True
            return True, self.positions[12]  # Return Position 13
        return False, self.positions[self.current_position_idx]

    def get_current_position(self) -> Position:
        """
        Get the current position in the cycle.
        
        Returns:
            Current Position object
        """
        return self.positions[self.current_position_idx]

    def apply_field_effects(self, phase_field, energy_field) -> None:
        """
        Apply field effects to all positions in the cycle.
        
        Args:
            phase_field: Phase field object
            energy_field: Energy field object
        """
        for position in self.positions:
            # Extract field values at position's spatial location
            coords = position.spatial_location
            
            # Convert coordinates to grid indices
            grid_size = phase_field.grid_size
            i = int((coords[0] + 5) * grid_size[0] / 10) % grid_size[0]
            j = int((coords[1] + 5) * grid_size[1] / 10) % grid_size[1]
            k = int((coords[2] + 5) * grid_size[2] / 10) % grid_size[2]
            
            # Extract field values
            phase_value = np.angle(phase_field.field[i, j, k])
            energy_value = energy_field.field[i, j, k]
            
            # Apply field effects to position
            position.phase = phase_value
            position.energy_level = 0.8 * position.energy_level + 0.2 * energy_value
            
    def find_resonant_positions(self) -> List[Tuple[int, int, float]]:
        """
        Find pairs of positions with strong resonance.
        
        Returns:
            List of (position1_num, position2_num, resonance_value) tuples
        """
        resonant_pairs = []
        
        # Check each pair of positions
        for i in range(len(self.positions)):
            for j in range(i+1, len(self.positions)):
                pos1 = self.positions[i]
                pos2 = self.positions[j]
                
                # Calculate resonance
                resonance = pos1.calculate_resonance_with(pos2)
                
                # If resonance is strong enough, add to the list
                if resonance > 0.7:
                    resonant_pairs.append((pos1.number, pos2.number, resonance))
                    
        return sorted(resonant_pairs, key=lambda x: x[2], reverse=True)

    def get_cycle_harmonic_signature(self) -> np.ndarray:
        """
        Calculate the combined harmonic signature of the whole cycle.
        
        Returns:
            NumPy array of signature values
        """
        # Initialize signature
        signature = np.zeros(13)
        
        # Combine signatures from all positions
        for position in self.positions:
            pos_signature = position.get_harmonic_signature()
            signature += pos_signature
            
        # Normalize
        if np.sum(signature) > 0:
            signature = signature / np.sum(signature)
            
        return signature
