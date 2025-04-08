"""
Torus module for the UTCHS framework.

This module implements the Torus class, which manages multiple scaled structures
and handles the seed-to-torus transition mechanism.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from utchs.core.structure import ScaledStructure

class Torus:
    """
    Represents a complete toroidal geometry composed of multiple scaled structures.
    """
    def __init__(self, id: int, num_structures: int = 7, seed_data: Optional[Dict] = None):
        """
        Initialize a torus with multiple scaled structures.

        Args:
            id: Torus identifier
            num_structures: Number of scaled structures (default: 7)
            seed_data: Optional data from Position 10 in previous torus
        """
        self.id = id
        self.structures = [ScaledStructure(i, id) for i in range(1, num_structures + 1)]
        self.current_structure_idx = 0
        self.completed = False
        self.seed_data = seed_data
        
        # Toroidal parameters
        self.major_radius = 2.0 + 0.5 * (id - 1)  # Increase with each torus
        self.minor_radius = 0.5 + 0.1 * (id - 1)  # Increase with each torus
        self.phase_rotation = (id - 1) * np.pi / 4  # Phase rotation per torus
        
        # Apply seed data if provided
        if seed_data:
            self._initialize_from_seed(seed_data)
            
        # Calculate torus properties based on its id
        self.energy_capacity = self._calculate_energy_capacity()
        self.phase_coherence = 1.0  # Initial perfect coherence
        self.stability_metric = 0.0  # Initial zero stability (will evolve)

    def _initialize_from_seed(self, seed_data: Dict) -> None:
        """Initialize the first position from seed data."""
        # Get first position in the first cycle of the first structure
        first_position = self.structures[0].cycles[0].positions[0]
        
        # Transfer appropriate properties from seed data to first position
        if 'energy_level' in seed_data:
            first_position.energy_level = seed_data['energy_level']
            
        if 'phase' in seed_data:
            first_position.phase = seed_data['phase']
            
        if 'spatial_location' in seed_data:
            # Scale and rotate the spatial location
            location = np.array(seed_data['spatial_location'])
            scale_factor = 0.618  # Golden ratio scaling
            rotation_angle = np.pi / 4  # 45 degree rotation
            
            # Apply scaling
            location = location * scale_factor
            
            # Apply rotation around z-axis
            cos_theta = np.cos(rotation_angle)
            sin_theta = np.sin(rotation_angle)
            rotation_matrix = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
            location = np.dot(rotation_matrix, location)
            
            first_position.spatial_location = location
            
        if 'harmonic_signature' in seed_data:
            # This would be used in a more complex initialization
            # that transfers the harmonic pattern from the seed
            pass

    def _calculate_energy_capacity(self) -> float:
        """
        Calculate the energy capacity of the torus.
        
        Returns:
            Energy capacity value
        """
        # Base capacity increases with torus id
        base_capacity = 100.0 * (1.0 + 0.2 * (self.id - 1))
        
        # Apply golden ratio modulation
        phi = (1 + np.sqrt(5)) / 2
        phi_factor = phi ** (self.id % 5)
        
        # Apply prime number amplification
        prime_factor = 1.0
        if self.id in [2, 3, 5, 7, 11, 13]:  # Prime torus ids
            prime_factor = 1.5
            
        return base_capacity * phi_factor * prime_factor

    def advance_structure(self) -> bool:
        """
        Advance to the next structure in the torus.
        
        Returns:
            True if torus is completed, False otherwise
        """
        self.current_structure_idx += 1
        if self.current_structure_idx >= len(self.structures):
            self.current_structure_idx = 0
            self.completed = True
            return True
        return False

    def get_current_structure(self) -> ScaledStructure:
        """Get the currently active structure."""
        return self.structures[self.current_structure_idx]

    def extract_seed(self) -> Dict:
        """
        Extract seed data from Position 10 of the last structure.
        
        Returns:
            Dictionary containing seed data
        """
        # Get Position 10 from the last structure
        last_structure = self.structures[-1]
        last_cycle = last_structure.cycles[-1]
        seed_position = last_cycle.positions[9]  # Position 10 (0-based index)
        
        # Extract relevant properties
        seed_data = {
            'energy_level': seed_position.energy_level,
            'phase': seed_position.phase,
            'spatial_location': seed_position.spatial_location.tolist(),
            'harmonic_signature': seed_position.harmonic_signature
        }
        
        return seed_data
        
    def _update_torus_metrics(self) -> None:
        """Update torus-wide metrics based on current state."""
        # Calculate phase coherence across all structures
        phase_values = []
        for structure in self.structures:
            for cycle in structure.cycles:
                for position in cycle.positions:
                    phase_values.append(position.phase)
                    
        # Calculate circular mean of phases
        phase_array = np.array(phase_values)
        mean_phase = np.arctan2(
            np.mean(np.sin(phase_array)),
            np.mean(np.cos(phase_array))
        )
        
        # Calculate coherence as inverse of phase dispersion
        phase_dispersion = np.mean(np.abs(phase_array - mean_phase))
        self.phase_coherence = 1.0 / (1.0 + phase_dispersion)
        
        # Update stability metric
        self.stability_metric = self.phase_coherence * (1.0 - 0.1 * (self.id - 1))

    def apply_field_effects(self, phase_field, energy_field) -> None:
        """
        Apply field effects to all positions in the torus.
        
        Args:
            phase_field: Phase field instance
            energy_field: Energy field instance
        """
        for structure in self.structures:
            for cycle in structure.cycles:
                for position in cycle.positions:
                    # Apply phase field effects
                    position.phase += phase_field.get_phase_effect(position.spatial_location)
                    
                    # Apply energy field effects
                    position.energy_level += energy_field.get_energy_effect(position.spatial_location)
                    
                    # Ensure energy level stays within bounds
                    position.energy_level = np.clip(position.energy_level, 0.0, self.energy_capacity)

    def get_energy_distribution(self) -> np.ndarray:
        """
        Get the energy distribution across all positions.
        
        Returns:
            Array of energy values
        """
        energy_values = []
        for structure in self.structures:
            for cycle in structure.cycles:
                for position in cycle.positions:
                    energy_values.append(position.energy_level)
                    
        return np.array(energy_values)
        
    def calculate_phase_patterns(self) -> Dict:
        """
        Calculate phase patterns across the torus.
        
        Returns:
            Dictionary containing phase pattern analysis
        """
        # Collect phase values and positions
        position_phases = []
        for structure in self.structures:
            for cycle in structure.cycles:
                for position in cycle.positions:
                    position_phases.append({
                        'phase': position.phase,
                        'position_number': position.number,
                        'spatial_location': position.spatial_location
                    })
                    
        # Sort by position number
        position_phases.sort(key=lambda x: x['position_number'])
        
        # Calculate phase differences between adjacent positions
        phase_differences = []
        for i in range(len(position_phases) - 1):
            phase_diff = position_phases[i+1]['phase'] - position_phases[i]['phase']
            # Normalize to [-pi, pi]
            phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
            phase_differences.append(phase_diff)
            
        # Calculate average phase difference
        avg_phase_diff = np.mean(phase_differences)
        
        # Calculate phase gradient
        phase_gradient = np.gradient([p['phase'] for p in position_phases])
        
        return {
            'position_phases': position_phases,
            'phase_differences': phase_differences,
            'average_phase_difference': avg_phase_diff,
            'phase_gradient': phase_gradient.tolist()
        }
    
    def get_serializable_state(self) -> Dict:
        """
        Get a serializable representation of the torus state.
        
        Returns:
            Dictionary with torus state
        """
        return {
            "id": self.id,
            "current_structure_idx": self.current_structure_idx,
            "completed": self.completed,
            "major_radius": self.major_radius,
            "minor_radius": self.minor_radius,
            "phase_rotation": self.phase_rotation,
            "energy_capacity": self.energy_capacity,
            "phase_coherence": self.phase_coherence,
            "stability_metric": self.stability_metric,
            "structure_states": [s.get_serializable_state() for s in self.structures]
        }
    
    def load_state(self, state: Dict) -> None:
        """
        Load torus state from a dictionary.
        
        Args:
            state: Dictionary with torus state
        """
        self.id = state["id"]
        self.current_structure_idx = state["current_structure_idx"]
        self.completed = state["completed"]
        self.major_radius = state["major_radius"]
        self.minor_radius = state["minor_radius"]
        self.phase_rotation = state["phase_rotation"]
        self.energy_capacity = state["energy_capacity"]
        self.phase_coherence = state["phase_coherence"]
        self.stability_metric = state["stability_metric"]
