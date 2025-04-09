"""
Phase field module for the UTCHS framework.

This module implements the PhaseField class, which manages phase dynamics and evolution
through Möbius transformations, as well as phase singularity detection.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import numpy.typing as npt

from ..core.position import Position
from .mobius import MobiusTransformation
from ..utils.logging_config import get_logger, FieldError, ConfigurationError
from ..utils.base_classes import FieldBase

# Get logger for this module
logger = get_logger(__name__)

class PhaseField(FieldBase):
    """
    Implements the phase field dynamics with Möbius transformations.

    The phase field evolves according to:
    φ(z, t+Δt) = (a(t)φ(z,t) + b(t))/(c(t)φ(z,t) + d(t))

    Attributes:
        grid_size (Tuple[int, ...]): Size of the phase field grid
        dx (float): Grid spacing
        field (npt.NDArray[np.complex128]): Complex-valued phase field
        phi_components (npt.NDArray[np.float64]): Torsion field components
        singularities (List[Dict]): List of phase singularity information
        history_length (int): Length of field history to maintain
        field_history (List[npt.NDArray[np.complex128]]): History of field states
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the phase field.

        Args:
            config: Configuration dictionary containing:
                - grid_size: Tuple of grid dimensions
                - grid_spacing: Spacing between grid points
                - initial_pattern: Pattern for field initialization
                - history_length: Number of historical states to maintain
                
        Raises:
            ConfigurationError: If required configuration parameters are missing or invalid
        """
        super().__init__(name="phase_field", config=config)
    
    def _pre_initialize(self) -> None:
        """Pre-initialization steps."""
        # Call parent pre_initialize to set up basic field attributes
        super()._pre_initialize()
        
        # Validate configuration early
        try:
            self._validate_config(self.config)
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            raise ConfigurationError(f"Invalid configuration: {str(e)}")
    
    def _initialize(self) -> None:
        """Main initialization steps."""
        # Initialize phase field (complex-valued scalar field)
        self.field = np.zeros(self.grid_size, dtype=complex)
        
        # Initialize torsion field components
        self.phi_components = np.zeros((3, *self.grid_size))
        
        # Track singularities
        self.singularities = []
        
        # Initialize with a specified pattern
        pattern = self.config.get('initial_pattern', 'random')
        try:
            self._initialize_field(pattern)
            self.logger.info(f"Phase field initialized with pattern: {pattern}")
        except Exception as e:
            self.logger.error(f"Error initializing field with pattern {pattern}: {str(e)}")
            # Continue with default field rather than failing completely
            self.logger.info("Continuing with default zero field")
    
    def _post_initialize(self) -> None:
        """Post-initialization steps."""
        # Store initial field in history
        self.update_history()
        self.logger.info(f"Phase field initialized with grid size {self.grid_size}")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_params = ['grid_size', 'grid_spacing']
        self.validate_config(required_params)
                
        # Validate grid size
        if not isinstance(config['grid_size'], (tuple, list)) or len(config['grid_size']) != 3:
            raise ConfigurationError("grid_size must be a tuple or list of length 3")
            
        # Validate grid spacing
        if not isinstance(config['grid_spacing'], (int, float)) or config['grid_spacing'] <= 0:
            raise ConfigurationError("grid_spacing must be a positive number")
            
        self.logger.debug("Phase field configuration validation successful")

    def _initialize_field(self, pattern: str) -> None:
        """
        Initialize the phase field with a specified pattern.

        Args:
            pattern: Initial pattern type ('random', 'vortex', 'spiral', etc.)
        """
        try:
            nx, ny, nz = self.grid_size
            x = np.linspace(-1, 1, nx)
            y = np.linspace(-1, 1, ny)
            z = np.linspace(-1, 1, nz)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            if pattern == 'random':
                # Random phases between 0 and 2π
                phases = 2 * np.pi * np.random.random(self.grid_size)
                amplitudes = np.ones(self.grid_size)
                self.field = amplitudes * np.exp(1j * phases)
                
            elif pattern == 'vortex':
                # Single vortex at the center
                R = np.sqrt(X**2 + Y**2 + Z**2)
                Theta = np.arctan2(Y, X)
                Phi = np.arccos(Z / (R + 1e-10))
                
                # Phase increases with azimuthal angle
                phases = Theta
                amplitudes = np.ones(self.grid_size) * np.exp(-R**2)
                self.field = amplitudes * np.exp(1j * phases)
                
            elif pattern == 'spiral':
                # Spiral pattern
                R = np.sqrt(X**2 + Y**2)
                Theta = np.arctan2(Y, X)
                
                # Phase is a function of both radius and angle
                phases = Theta + 5 * R
                amplitudes = np.ones(self.grid_size) * np.exp(-0.5 * R**2)
                self.field = amplitudes * np.exp(1j * phases)
                
            elif pattern == 'toroidal':
                # Toroidal pattern
                # Major and minor radii
                R_major = 0.5
                R_minor = 0.2
                
                # Calculate distance from torus ring
                R_ring = np.sqrt(X**2 + Y**2) - R_major
                R_torus = np.sqrt(R_ring**2 + Z**2)
                
                # Phase based on poloidal and toroidal angles
                Theta_toroidal = np.arctan2(Y, X)
                Theta_poloidal = np.arctan2(Z, R_ring)
                
                phases = 2*Theta_toroidal + 3*Theta_poloidal
                amplitudes = np.exp(-(R_torus - R_minor)**2 / (0.1**2))
                
                self.field = amplitudes * np.exp(1j * phases)
                
            else:
                # Default to constant phase
                self.field = np.ones(self.grid_size)
                
        except Exception as e:
            self.logger.error(f"Error initializing field with pattern {pattern}: {str(e)}")
            raise RuntimeError(f"Field initialization error: {str(e)}")

    def update(self, position: Position, mobius_params: Dict[str, complex]) -> None:
        """
        Update the phase field using a Möbius transformation.

        Args:
            position: Current position in the UTCHS hierarchy
            mobius_params: Parameters for the Möbius transformation
        """
        try:
            a, b, c, d = mobius_params['a'], mobius_params['b'], mobius_params['c'], mobius_params['d']
            
            # Create Möbius transformation
            mobius = MobiusTransformation(a, b, c, d)
            
            # Apply transformation to the entire field
            self.field = mobius.transform(self.field)
            
            # Update phase derivative components for torsion calculation
            self._update_phi_components()
            
            # Find phase singularities
            self._find_singularities()
            
            # Update field history
            self.update_history()
            
            self.logger.debug(f"Phase field updated for position {position.number}")
            
        except Exception as e:
            self.logger.error(f"Error updating phase field: {str(e)}")
            raise RuntimeError(f"Field update error: {str(e)}")

    def _update_phi_components(self) -> None:
        """Update the phi components for torsion field calculation."""
        try:
            # Real part (x-component)
            self.phi_components[0] = np.real(self.field)
            
            # Imaginary part (y-component)
            self.phi_components[1] = np.imag(self.field)
            
            # Phase angle (z-component)
            self.phi_components[2] = np.angle(self.field)
            
        except Exception as e:
            self.logger.error(f"Error updating phi components: {str(e)}")
            self.logger.warning("Continuing with previous phi components")

    def _find_singularities(self) -> None:
        """Find phase singularities in the field."""
        try:
            nx, ny, nz = self.grid_size
            singularities = []
            
            # Loop through interior positions
            for x_index in range(1, nx-1):
                for y_index in range(1, ny-1):
                    for z_index in range(1, nz-1):
                        # Check if amplitude is zero or near-zero (potential singularity)
                        if abs(self.field[x_index, y_index, z_index]) < 1e-5:
                            # Check phase circulation around this position
                            phase_circulation = self._calculate_phase_circulation((x_index, y_index, z_index))
                            
                            # If circulation is approximately 2π or -2π, it's a singularity
                            if abs(abs(phase_circulation) - 2*np.pi) < 0.1:
                                # Record position and topological charge
                                charge = 1 if phase_circulation > 0 else -1
                                singularities.append({
                                    'position': (x_index, y_index, z_index),
                                    'charge': charge
                                })
            
            self.singularities = singularities
            
            if singularities:
                self.logger.debug(f"Found {len(singularities)} phase singularities")
                
        except Exception as e:
            self.logger.error(f"Error finding singularities: {str(e)}")
            self.logger.warning("Continuing with previous singularities")

    def _calculate_phase_circulation(self, position: Tuple[int, int, int]) -> float:
        """
        Calculate phase circulation around a field position.

        Args:
            position: Grid indices (x_index, y_index, z_index) in the field

        Returns:
            Phase circulation value
        """
        x_index, y_index, z_index = position
        
        # Create a loop of field positions around the target position
        loop_positions = [
            (x_index+1, y_index, z_index), 
            (x_index+1, y_index+1, z_index),
            (x_index, y_index+1, z_index), 
            (x_index-1, y_index+1, z_index),
            (x_index-1, y_index, z_index), 
            (x_index-1, y_index-1, z_index),
            (x_index, y_index-1, z_index), 
            (x_index+1, y_index-1, z_index),
            (x_index+1, y_index, z_index)
        ]
        
        # Calculate phase differences around the loop
        circulation = 0.0
        for idx in range(len(loop_positions)-1):
            current_position = loop_positions[idx]
            next_position = loop_positions[idx+1]
            
            current_phase = np.angle(self.field[current_position])
            next_phase = np.angle(self.field[next_position])
            
            # Calculate phase difference, handling 2π wraparound
            phase_difference = (next_phase - current_phase + np.pi) % (2*np.pi) - np.pi
            circulation += phase_difference
            
        return circulation

    def calculate_total_energy(self) -> float:
        """
        Calculate the total energy in the phase field.

        Returns:
            Total energy value
        """
        # Energy is proportional to the squared amplitude
        return np.sum(np.abs(self.field)**2)

    def calculate_phase_gradient(self) -> np.ndarray:
        """
        Calculate the gradient of the phase field.
        
        Returns:
            Phase gradient as a vector field of shape (3, nx, ny, nz)
        """
        nx, ny, nz = self.grid_size
        phase = np.angle(self.field)
        gradient = np.zeros((3, nx, ny, nz), dtype=float)
        
        # Calculate gradients using finite differences
        # x-component
        gradient[0, 1:-1, :, :] = (phase[2:, :, :] - phase[:-2, :, :]) / (2 * self.dx)
        gradient[0, 0, :, :] = (phase[1, :, :] - phase[0, :, :]) / self.dx
        gradient[0, -1, :, :] = (phase[-1, :, :] - phase[-2, :, :]) / self.dx
        
        # y-component
        gradient[1, :, 1:-1, :] = (phase[:, 2:, :] - phase[:, :-2, :]) / (2 * self.dx)
        gradient[1, :, 0, :] = (phase[:, 1, :] - phase[:, 0, :]) / self.dx
        gradient[1, :, -1, :] = (phase[:, -1, :] - phase[:, -2, :]) / self.dx
        
        # z-component
        gradient[2, :, :, 1:-1] = (phase[:, :, 2:] - phase[:, :, :-2]) / (2 * self.dx)
        gradient[2, :, :, 0] = (phase[:, :, 1] - phase[:, :, 0]) / self.dx
        gradient[2, :, :, -1] = (phase[:, :, -1] - phase[:, :, -2]) / self.dx
        
        return gradient

    def get_phase_effect(self, spatial_location: tuple) -> float:
        """
        Calculate the phase effect at a specific spatial location.
        
        Args:
            spatial_location: Spatial coordinates (x, y, z) in the field
            
        Returns:
            Phase effect value
        """
        # Convert spatial location to grid indices
        nx, ny, nz = self.grid_size
        x_range = np.linspace(-1, 1, nx)
        y_range = np.linspace(-1, 1, ny)
        z_range = np.linspace(-1, 1, nz)
        
        # Find nearest grid points
        x, y, z = spatial_location
        x_idx = min(max(0, np.searchsorted(x_range, x)), nx-1)
        y_idx = min(max(0, np.searchsorted(y_range, y)), ny-1)
        z_idx = min(max(0, np.searchsorted(z_range, z)), nz-1)
        
        # Get phase at this location
        phase = np.angle(self.field[x_idx, y_idx, z_idx])
        
        # Calculate a phase effect based on the current phase
        # For now, using a simple sinusoidal effect
        phase_effect = 0.1 * np.sin(phase)
        
        return phase_effect

    def apply_physical_constraints(self) -> None:
        """
        Apply physical constraints to the field to ensure stability.
        
        This can include normalization, amplitude bounds, or other constraints.
        """
        # Normalize amplitude if it exceeds threshold
        max_amplitude = np.max(np.abs(self.field))
        if max_amplitude > 10.0:
            self.field = self.field * (10.0 / max_amplitude)

    def get_serializable_state(self) -> Dict:
        """
        Get a serializable representation of the field state.

        Returns:
            Dictionary with field state
        """
        state = super().get_serializable_state()
        
        # Add field-specific data in serializable format
        field_data = {
            'field_real': np.real(self.field).tolist(),
            'field_imag': np.imag(self.field).tolist(),
            'singularities': self.singularities
        }
        
        state.update(field_data)
        return state

    def load_state(self, state: Dict) -> None:
        """
        Load field state from a dictionary.

        Args:
            state: Dictionary with field state
        """
        if 'field_real' in state and 'field_imag' in state:
            real_part = np.array(state['field_real'])
            imag_part = np.array(state['field_imag'])
            self.field = real_part + 1j * imag_part
            
        if 'singularities' in state:
            self.singularities = state['singularities']
            
        # Update history
        self.update_history()
