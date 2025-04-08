"""
Structure module for the UTCHS framework.

This module implements the ScaledStructure class, which manages multiple cycles
within the hierarchical UTCHS framework.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from utchs.core.cycle import Cycle

class ScaledStructure:
    """
    Represents a scaled 13D structure composed of multiple cycles.
    """
    def __init__(self, id: int, torus_id: int, num_cycles: int = 7):
        """
        Initialize a scaled structure within a torus.

        Args:
            id: Structure identifier within torus
            torus_id: Parent torus identifier
            num_cycles: Number of cycles in this structure (default: 7)
        """
        self.id = id
        self.torus_id = torus_id
        self.cycles = [Cycle(i, torus_id, id) for i in range(1, num_cycles + 1)]
        self.current_cycle_idx = 0
        self.completed = False
        
        # Golden ratio scaling factor
        self.phi = (1 + np.sqrt(5)) / 2
        
        # Structure properties
        self.scale_factor = self._calculate_scale_factor()
        self.rotation_factor = self._calculate_rotation_factor()
        self.energy_profile = self._calculate_energy_profile()
        
    def _calculate_scale_factor(self) -> float:
        """
        Calculate the scale factor for this structure based on golden ratio.
        
        Returns:
            Scale factor value
        """
        # Apply φ-based scaling based on structure id
        return self.phi ** (1 - self.id % 7)
        
    def _calculate_rotation_factor(self) -> float:
        """
        Calculate the rotation factor for this structure.
        
        Returns:
            Rotation factor in radians
        """
        # Use golden angle (137.5°) scaled by structure id
        golden_angle = np.pi * (3 - np.sqrt(5))  # ~137.5 degrees in radians
        return golden_angle * (self.id / self.phi)
        
    def _calculate_energy_profile(self) -> np.ndarray:
        """
        Calculate the energy distribution profile for this structure.
        
        Returns:
            Energy profile array
        """
        # Create energy profile based on structure id and φ-scaling
        profile = np.zeros(13)
        
        for i in range(13):
            # Position-based factor
            pos_factor = 1.0 + 0.1 * np.sin(i * np.pi / 6.5)
            
            # Prime-number amplification
            if i+1 in [1, 2, 3, 5, 7, 11, 13]:  # Prime positions
                pos_factor *= 1.2
                
            # Structure-specific factor
            struct_factor = 1.0 + 0.15 * np.sin(self.id * np.pi / 3.5)
            
            # Combine factors
            profile[i] = pos_factor * struct_factor
        
        # Normalize
        profile = profile / np.sum(profile)
        
        return profile

    def advance_cycle(self) -> bool:
        """
        Advance to the next cycle in the structure.

        Returns:
            True if structure is completed, False otherwise
        """
        self.current_cycle_idx += 1
        if self.current_cycle_idx >= len(self.cycles):
            self.completed = True
            return True
        return False

    def get_current_cycle(self) -> Cycle:
        """Get the current cycle."""
        return self.cycles[self.current_cycle_idx]

    def get_scaled_harmonics(self) -> np.ndarray:
        """
        Calculate the harmonic pattern across all cycles.
        
        Returns:
            Harmonic pattern array
        """
        # Initialize harmonic pattern
        harmonics = np.zeros(13)
        
        # Combine harmonic signatures from all cycles, applying φ-scaling
        for i, cycle in enumerate(self.cycles):
            cycle_signature = cycle.get_cycle_harmonic_signature()
            scale = self.phi ** (-(i+1))  # Progressive φ-scaling
            harmonics += cycle_signature * scale
            
        # Apply structure-specific phase shift
        phase_shift = 2 * np.pi * (self.id / 7)
        shifted_harmonics = np.zeros_like(harmonics)
        
        for i in range(13):
            shifted_idx = int((i + phase_shift * 13 / (2 * np.pi)) % 13)
            shifted_harmonics[shifted_idx] = harmonics[i]
            
        # Normalize
        if np.sum(shifted_harmonics) > 0:
            shifted_harmonics = shifted_harmonics / np.sum(shifted_harmonics)
            
        return shifted_harmonics

    def apply_field_effects(self, phase_field, energy_field) -> None:
        """
        Apply field effects to all cycles in the structure.
        
        Args:
            phase_field: Phase field object
            energy_field: Energy field object
        """
        for cycle in self.cycles:
            cycle.apply_field_effects(phase_field, energy_field)
            
    def get_position_network(self) -> Dict:
        """
        Get the network of positions and their relationships.
        
        Returns:
            Dictionary describing the position network
        """
        network = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes for all positions
        for cycle in self.cycles:
            for position in cycle.positions:
                node = {
                    "id": f"C{cycle.id}_P{position.number}",
                    "cycle": cycle.id,
                    "position": position.number,
                    "role": position.role,
                    "energy": position.energy_level,
                    "phase": position.phase,
                    "coordinates": position.spatial_location.tolist()
                }
                network["nodes"].append(node)
        
        # Add edges for connections
        edge_id = 0
        for cycle in self.cycles:
            # Connect positions within cycles
            for i in range(len(cycle.positions)):
                pos1 = cycle.positions[i]
                pos2 = cycle.positions[(i+1) % 13]  # Next position in cycle
                
                edge = {
                    "id": f"E{edge_id}",
                    "source": f"C{cycle.id}_P{pos1.number}",
                    "target": f"C{cycle.id}_P{pos2.number}",
                    "type": "sequential",
                    "weight": 1.0
                }
                network["edges"].append(edge)
                edge_id += 1
                
            # Connect resonant positions
            resonant_pairs = cycle.find_resonant_positions()
            for pos1_num, pos2_num, resonance in resonant_pairs:
                edge = {
                    "id": f"E{edge_id}",
                    "source": f"C{cycle.id}_P{pos1_num}",
                    "target": f"C{cycle.id}_P{pos2_num}",
                    "type": "resonant",
                    "weight": resonance
                }
                network["edges"].append(edge)
                edge_id += 1
                
        # Connect positions between adjacent cycles
        for i in range(len(self.cycles)-1):
            cycle1 = self.cycles[i]
            cycle2 = self.cycles[i+1]
            
            # Connect Position 13 of cycle1 to Position 1 of cycle2
            edge = {
                "id": f"E{edge_id}",
                "source": f"C{cycle1.id}_P13",
                "target": f"C{cycle2.id}_P1",
                "type": "cycle_transition",
                "weight": 1.5
            }
            network["edges"].append(edge)
            edge_id += 1
            
            # Connect Position 10 of cycle1 to Position 1 of cycle2 (seed connection)
            edge = {
                "id": f"E{edge_id}",
                "source": f"C{cycle1.id}_P10",
                "target": f"C{cycle2.id}_P1",
                "type": "seed",
                "weight": 1.2
            }
            network["edges"].append(edge)
            edge_id += 1
            
        return network
        
    def analyze_vortex_points(self) -> Dict:
        """
        Analyze the vortex points (positions 3, 6, 9) across all cycles.
        
        Returns:
            Dictionary with vortex point analysis
        """
        analysis = {
            "vortex_points": [],
            "energy_ratio": 0,
            "phase_coherence": 0,
            "stability_factor": 0
        }
        
        # Collect all vortex points
        vortex_positions = []
        for cycle in self.cycles:
            for position in cycle.positions:
                if position.number in [3, 6, 9]:  # Vortex points
                    vortex_positions.append(position)
                    
                    analysis["vortex_points"].append({
                        "cycle": cycle.id,
                        "position": position.number,
                        "energy": position.energy_level,
                        "phase": position.phase
                    })
        
        # Skip further analysis if no vortex points found
        if not vortex_positions:
            return analysis
            
        # Calculate energy ratio (vortex energy / total energy)
        vortex_energy = sum(pos.energy_level for pos in vortex_positions)
        total_energy = sum(pos.energy_level for cycle in self.cycles for pos in cycle.positions)
        analysis["energy_ratio"] = vortex_energy / total_energy if total_energy > 0 else 0
        
        # Calculate phase coherence
        phases = [pos.phase for pos in vortex_positions]
        # Using circular statistics for phase coherence
        sin_sum = sum(np.sin(phases))
        cos_sum = sum(np.cos(phases))
        r = np.sqrt(sin_sum**2 + cos_sum**2) / len(phases)
        analysis["phase_coherence"] = r
        
        # Calculate stability factor
        stability_factor = analysis["energy_ratio"] * analysis["phase_coherence"]
        # Apply 3-6-9 ratio factor
        pos3_count = sum(1 for pos in vortex_positions if pos.number == 3)
        pos6_count = sum(1 for pos in vortex_positions if pos.number == 6)
        pos9_count = sum(1 for pos in vortex_positions if pos.number == 9)
        
        # Ideal ratio is 1:1:1
        ratio_balance = 1.0 - 0.5 * (
            abs(pos3_count - pos6_count) / max(1, pos3_count + pos6_count) +
            abs(pos6_count - pos9_count) / max(1, pos6_count + pos9_count) +
            abs(pos9_count - pos3_count) / max(1, pos9_count + pos3_count)
        )
        
        analysis["stability_factor"] = stability_factor * ratio_balance
        
        return analysis
        
    def get_serializable_state(self) -> Dict:
        """
        Get a serializable representation of the structure state.
        
        Returns:
            Dictionary with structure state
        """
        return {
            "id": self.id,
            "torus_id": self.torus_id,
            "current_cycle_idx": self.current_cycle_idx,
            "completed": self.completed,
            "scale_factor": self.scale_factor,
            "rotation_factor": self.rotation_factor,
            "energy_profile": self.energy_profile.tolist()
        }
    
    def load_state(self, state: Dict) -> None:
        """
        Load structure state from a dictionary.
        
        Args:
            state: Dictionary with structure state
        """
        self.id = state["id"]
        self.torus_id = state["torus_id"]
        self.current_cycle_idx = state["current_cycle_idx"]
        self.completed = state["completed"]
        self.scale_factor = state["scale_factor"]
        self.rotation_factor = state["rotation_factor"]
        self.energy_profile = np.array(state["energy_profile"])