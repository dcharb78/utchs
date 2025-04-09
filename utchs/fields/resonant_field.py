"""
Resonant field module for the UTCHS framework.

This module implements the ResonantField class, which models resonance patterns and
harmonic alignments based on the OmniLens Framework concept.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union, Any
from ..math.mobius import MobiusNode
from ..utils.validation_registry import get_base_resonant_frequencies, validate_resonant_frequency

class ResonantField:
    """
    Models resonance patterns and harmonic alignments within the UTCHS framework.
    
    A ResonantField represents a field of resonant harmonics that can interact with
    phase fields and Möbius nodes, implementing the resonance pattern concepts
    from the OmniLens Möbius Framework.
    
    Attributes:
        tuning: Base resonance frequency
        dimensions: Field dimensionality
        resolution: Grid resolution per dimension
        field: Numpy array representing the resonance field
    """
    
    def __init__(self, tuning: Optional[float] = None, dimensions: int = 3, resolution: int = 32):
        """
        Initialize a resonant field with specified tuning.
        
        Args:
            tuning: Base resonance frequency (default: determined by system properties)
            dimensions: Number of field dimensions (default: 3)
            resolution: Grid resolution per dimension (default: 32)
        """
        # If tuning is not provided, use the primary resonant frequency from validation framework
        if tuning is None:
            self.tuning = get_base_resonant_frequencies()[0]  # Primary frequency (144000)
        else:
            # Validate the provided tuning against known resonant frequencies
            if validate_resonant_frequency(tuning):
                self.tuning = tuning
            else:
                # If invalid, use the closest valid frequency
                base_freqs = get_base_resonant_frequencies()
                closest_freq = min(base_freqs, key=lambda f: abs(f - tuning))
                self.tuning = closest_freq
        
        self.dimensions = dimensions
        self.resolution = resolution
        
        # Initialize field grid
        shape = tuple([resolution] * dimensions)
        self.field = np.zeros(shape, dtype=float)
        
        # Initialize with default resonance pattern
        self._initialize_resonance_pattern()
        
    def _initialize_resonance_pattern(self) -> None:
        """Initialize the field with a default resonance pattern."""
        # Create coordinate meshgrid for field initialization
        coords = []
        for d in range(self.dimensions):
            coord = np.linspace(-1, 1, self.resolution)
            coords.append(coord)
            
        # Create meshgrid for coordinates
        grid_coords = np.meshgrid(*coords, indexing='ij')
        
        # Calculate distance from center for radial patterns
        r_squared = np.zeros_like(self.field)
        for coord in grid_coords:
            r_squared += coord**2
        r = np.sqrt(r_squared)
        
        # Toroidal resonance pattern (basic standing wave pattern)
        # Use a radial standing wave pattern with frequency related to tuning
        # Scale by 10000 to convert to reasonable spatial frequency
        frequency = self.tuning / 10000  # Scale to reasonable range
        self.field = np.cos(frequency * r) * np.exp(-0.5 * r**2)
        
    def harmonic_alignment(self, phase: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate harmonic alignment for a given phase.
        
        Implements a bell-like alignment profile where maximum alignment occurs
        at phase = 180 degrees (π radians) and minimum at 0/360 degrees (0/2π radians).
        
        Args:
            phase: Phase values in radians
            
        Returns:
            Alignment values in range [0, 1]
        """
        # Normalize phase to [0, 2π]
        normalized_phase = phase % (2 * np.pi)
        
        # Bell-like alignment profile
        return 1 - abs(normalized_phase - np.pi) / np.pi
    
    def apply_to_node(self, node: MobiusNode, position: Tuple[float, ...]) -> float:
        """
        Apply field effects to a MobiusNode at a specific position.
        
        Args:
            node: Target MobiusNode to affect
            position: Normalized position coordinates in field space
            
        Returns:
            Resulting coherence value
        """
        # Convert normalized position to grid indices
        indices = []
        for pos, dim in zip(position, range(self.dimensions)):
            # Clamp position to [-1, 1] range
            clamped_pos = max(-1, min(1, pos))
            # Convert to index
            idx = int((clamped_pos + 1) / 2 * (self.resolution - 1))
            indices.append(idx)
            
        # Get field value at position
        field_value = self.field[tuple(indices)]
        
        # Calculate harmonic alignment
        alignment = self.harmonic_alignment(node.phase)
        
        # Apply field effects to node
        coherence = node.recurse(field_value * alignment)
        
        return coherence
    
    def resonance_scan(self, phase_field: np.ndarray) -> np.ndarray:
        """
        Calculate resonance values for an entire phase field.
        
        Args:
            phase_field: Field of phase values in radians
            
        Returns:
            Field of resonance values
        """
        # Calculate alignment for each phase value
        alignment = self.harmonic_alignment(phase_field)
        
        # Scale by field values (using first dimension projection for simplicity)
        # In a more complex implementation, you would map the phase_field dimensions to
        # the resonant field dimensions appropriately
        field_projection = self.field
        if field_projection.shape != phase_field.shape:
            # Use the first 2D slice if dimensions don't match
            if self.dimensions >= 3:
                field_projection = self.field[0]
                
            # Broadcast to match phase field if needed
            if field_projection.shape != phase_field.shape:
                # Resize or reproject as needed - simplified here for example
                field_projection = np.ones_like(phase_field) * np.mean(field_projection)
        
        # Calculate resonance
        resonance = alignment * field_projection
        
        return resonance
    
    def tune_to_frequency(self, frequency: float) -> None:
        """
        Retune the field to a different base frequency.
        
        Args:
            frequency: New base resonance frequency
        """
        # Store original pattern to maintain pattern shape
        original_pattern = self.field.copy()
        
        # Validate the new frequency
        if not validate_resonant_frequency(frequency):
            # If invalid, find the closest valid frequency
            base_freqs = get_base_resonant_frequencies()
            frequency = min(base_freqs, key=lambda f: abs(f - frequency))
        
        # Update tuning
        ratio = frequency / self.tuning
        self.tuning = frequency
        
        # Reinitialize with scaled frequency
        self._initialize_resonance_pattern()
        
        # Blend with original pattern for smooth transition
        self.field = 0.7 * self.field + 0.3 * original_pattern
    
    def create_harmonic_overlay(self, harmonic_ratio: float) -> 'ResonantField':
        """
        Create a harmonic overlay field at specified harmonic ratio.
        
        Args:
            harmonic_ratio: Ratio for the harmonic overlay (e.g., 1.5 for perfect fifth)
            
        Returns:
            New ResonantField at the harmonic frequency
        """
        # Calculate new frequency
        new_frequency = self.tuning * harmonic_ratio
        
        # Create new field at harmonic frequency
        harmonic_field = ResonantField(
            tuning=new_frequency,
            dimensions=self.dimensions,
            resolution=self.resolution
        )
        
        return harmonic_field
        
    @classmethod
    def detect_resonant_tuning(cls, phase_values: list, dimensions: int = 3) -> float:
        """
        Detect the optimal resonant tuning frequency from phase value history.
        
        This method analyzes phase evolution to identify natural resonance points
        rather than using hardcoded values.
        
        Args:
            phase_values: List of phase values from system evolution
            dimensions: Field dimensions to use
            
        Returns:
            Detected resonant tuning frequency
        """
        from ..utils.validation_registry import detect_phase_lock_point
        
        # Detect phase lock point
        resonant_freq = detect_phase_lock_point(phase_values)
        
        if resonant_freq is None:
            # If no clear lock point detected, use the primary reference frequency
            resonant_freq = get_base_resonant_frequencies()[0]
            
        return resonant_freq 