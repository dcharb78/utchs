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

# Get logger for this module
logger = get_logger(__name__)

class PhaseField:
    """
    Implements the phase field dynamics with Möbius transformations.

    The phase field evolves according to:
    φ(z, t+Δt) = (a(t)φ(z,t) + b(t))/(c(t)φ(z,t) + d(t))

    Attributes:
        grid_size (Tuple[int, ...]): Size of the phase field grid
        dx (float): Grid spacing
        field (npt.NDArray[np.complex128]): Complex-valued phase field
        phi_components (npt.NDArray[np.float64]): Torsion field components
        singularities (List[Tuple[int, int, int]]): List of phase singularity coordinates
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
        try:
            # Validate configuration
            self._validate_config(config)
            
            self.grid_size = tuple(config.get('grid_size', (50, 50, 50)))
            self.dx = config.get('grid_spacing', 0.1)

            # Initialize phase field (complex-valued scalar field)
            self.field = np.zeros(self.grid_size, dtype=complex)
            
            # Initialize with a specified pattern
            self._initialize_field(config.get('initial_pattern', 'random'))
            
            # Initialize torsion field components
            self.phi_components = np.zeros((3, *self.grid_size))
            
            # Track singularities
            self.singularities = []
            
            # Field history for analysis
            self.history_length = config.get('history_length', 10)
            self.field_history = []
            
            logger.info(f"Phase field initialized with grid size {self.grid_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize phase field: {str(e)}")
            raise FieldError(f"Phase field initialization failed: {str(e)}") from e

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_params = ['grid_size', 'grid_spacing']
        for param in required_params:
            if param not in config:
                raise ConfigurationError(f"Missing required configuration parameter: {param}")
                
        # Validate grid size
        if not isinstance(config['grid_size'], (tuple, list)) or len(config['grid_size']) != 3:
            raise ConfigurationError("grid_size must be a tuple or list of length 3")
            
        # Validate grid spacing
        if not isinstance(config['grid_spacing'], (int, float)) or config['grid_spacing'] <= 0:
            raise ConfigurationError("grid_spacing must be a positive number")
            
        logger.debug("Phase field configuration validation successful")

    def _initialize_field(self, pattern: str) -> None:
        """
        Initialize the phase field with a specified pattern.

        Args:
            pattern: Initial pattern type ('random', 'vortex', 'spiral', etc.)
            
        Raises:
            FieldError: If field initialization fails
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
                
            # Store initial field in history
            self._update_history()
            
            logger.info(f"Phase field initialized with pattern: {pattern}")
            
        except Exception as e:
            logger.error(f"Error initializing phase field with pattern {pattern}: {str(e)}")
            raise FieldError(f"Phase field initialization failed: {str(e)}") from e

    def update(self, position: Position, mobius_params: Dict[str, complex]) -> None:
        """
        Update the phase field using a Möbius transformation.

        Args:
            position: Current position in the UTCHS hierarchy
            mobius_params: Parameters for the Möbius transformation
            
        Raises:
            FieldError: If field update fails
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
            self._update_history()
            
            logger.debug(f"Phase field updated for position {position.number}")
            
        except Exception as e:
            logger.error(f"Error updating phase field: {str(e)}")
            raise FieldError(f"Phase field update failed: {str(e)}") from e

    def _update_phi_components(self) -> None:
        """
        Update the phi components for torsion field calculation.
        
        Raises:
            FieldError: If component update fails
        """
        try:
            # Real part (x-component)
            self.phi_components[0] = np.real(self.field)
            
            # Imaginary part (y-component)
            self.phi_components[1] = np.imag(self.field)
            
            # Phase angle (z-component)
            self.phi_components[2] = np.angle(self.field)
            
        except Exception as e:
            logger.error(f"Error updating phi components: {str(e)}")
            raise FieldError(f"Phi component update failed: {str(e)}") from e

    def _update_history(self) -> None:
        """
        Update field history.
        
        Raises:
            FieldError: If history update fails
        """
        try:
            self.field_history.append(self.field.copy())
            
            # Limit history length
            while len(self.field_history) > self.history_length:
                self.field_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error updating field history: {str(e)}")
            raise FieldError(f"Field history update failed: {str(e)}") from e

    def _find_singularities(self) -> None:
        """
        Find phase singularities in the field.
        
        Raises:
            FieldError: If singularity detection fails
        """
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
                            phase_circulation = self._calculate_phase_circulation(x_index, y_index, z_index)
                            
                            # If circulation is approximately 2π or -2π, it's a singularity
                            if abs(abs(phase_circulation) - 2*np.pi) < 0.1:
                                # Record position and topological charge
                                charge = 1 if phase_circulation > 0 else -1
                                singularities.append({
                                    'field_position': (x_index, y_index, z_index),
                                    'charge': charge
                                })
            
            self.singularities = singularities
            
            if singularities:
                logger.debug(f"Found {len(singularities)} phase singularities")
                
        except Exception as e:
            logger.error(f"Error finding singularities: {str(e)}")
            raise FieldError(f"Singularity detection failed: {str(e)}") from e

    def _calculate_phase_circulation(self, x_index: int, y_index: int, z_index: int) -> float:
        """
        Calculate phase circulation around a field position.

        Args:
            x_index, y_index, z_index: Grid indices in the field

        Returns:
            Phase circulation value
        """
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
        return {
            'field_real': np.real(self.field),
            'field_imag': np.imag(self.field),
            'singularities': self.singularities
        }

    def load_state(self, state: Dict) -> None:
        """
        Load field state from a dictionary.

        Args:
            state: Dictionary with field state
        """
        self.field = state['field_real'] + 1j * state['field_imag']
        self.singularities = state['singularities']
        
        # Update phi components
        self._update_phi_components()
        
        # Reset history with current state
        self.field_history = [self.field.copy()]
