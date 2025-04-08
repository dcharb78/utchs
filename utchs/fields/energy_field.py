"""
Energy field module for the UTCHS framework.

This module implements the EnergyField class, which models field-based feedback energy
with non-local connections across the system.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

class EnergyField:
    """
    Implements field-based feedback energy with non-local connections.

    The energy field is described by:
    E(z, t) = E₀(z, t) + ∫ K(z, z') · E(z', t-τ(z, z')) dz'
    """
    def __init__(self, config: Dict):
        """
        Initialize the energy field.

        Args:
            config: Configuration dictionary
        """
        self.grid_size = tuple(config.get('grid_size', (50, 50, 50)))
        self.dx = config.get('grid_spacing', 0.1)
        
        # Initialize energy field
        self.field = np.zeros(self.grid_size, dtype=float)
        
        # Base energy field
        self.base_field = np.zeros(self.grid_size, dtype=float)
        
        # History of energy field for delay calculations
        self.history_length = config.get('history_length', 10)
        self.history = deque(maxlen=self.history_length)
        self.history.append(self.field.copy())
        
        # Kernel parameters
        self.alpha = config.get('kernel_decay_exponent', 2.0)  # Decay exponent
        self.kappa = config.get('kernel_coupling_strength', 0.1)  # Coupling strength
        self.phi_coupling = config.get('phi_coupling', 0.2)  # Phase coupling strength
        
        # Field metrics
        self.total_energy = 0.0
        self.peak_energy = 0.0
        self.energy_gradient_magnitude = 0.0
        
        # Golden ratio factor (φ) for scaling
        self.phi = (1 + np.sqrt(5)) / 2
        
        # Initialize with a base energy pattern
        self._initialize_field(config.get('initial_energy_pattern', 'gaussian'))

    def _initialize_field(self, pattern: str) -> None:
        """
        Initialize the energy field with a specific pattern.
        
        Args:
            pattern: Initial pattern type ('gaussian', 'toroidal', etc.)
        """
        nx, ny, nz = self.grid_size
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(-1, 1, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        if pattern == 'gaussian':
            # Gaussian energy distribution centered at origin
            R = np.sqrt(X**2 + Y**2 + Z**2)
            self.base_field = np.exp(-R**2 / 0.2**2)
            
        elif pattern == 'toroidal':
            # Toroidal energy distribution
            # Major and minor radii
            R_major = 0.5
            R_minor = 0.2
            
            # Calculate distance from torus ring
            R_ring = np.sqrt(X**2 + Y**2) - R_major
            R_torus = np.sqrt(R_ring**2 + Z**2)
            
            self.base_field = np.exp(-(R_torus - R_minor)**2 / (0.1**2))
            
        elif pattern == 'vortex':
            # Vortex-like energy distribution
            R_xy = np.sqrt(X**2 + Y**2)
            Theta = np.arctan2(Y, X)
            
            # Combine radial decay with angular modulation
            self.base_field = np.exp(-R_xy**2 / 0.3**2) * (1 + 0.3 * np.sin(3 * Theta)) * np.exp(-Z**2 / 0.5**2)
            
        elif pattern == 'fibonacci':
            # Fibonacci spiral energy pattern
            R = np.sqrt(X**2 + Y**2 + Z**2)
            Theta = np.arctan2(Y, X)
            Phi = np.arccos(Z / (R + 1e-10))
            
            # Golden angle for spiral
            golden_angle = np.pi * (3 - np.sqrt(5))
            
            # Create fibonaci-like spiral pattern
            spiral_factor = (Theta / golden_angle) % 1
            self.base_field = np.exp(-R**2 / 0.4**2) * (0.5 + 0.5 * np.sin(spiral_factor * 2 * np.pi)**2)
            
        else:
            # Default to uniform energy
            self.base_field = np.ones(self.grid_size) * 0.1
            
        # Copy to main field
        self.field = self.base_field.copy()
        
        # Update history
        self.history.clear()
        self.history.append(self.field.copy())
        
        # Calculate metrics
        self._update_metrics()

    def update(self, phase_field) -> None:
        """
        Update the energy field based on the phase field.

        Args:
            phase_field: Current phase field
        """
        # Update base energy field from phase field
        self._update_base_field(phase_field)
        
        # Add non-local feedback contribution
        self._add_feedback(phase_field)
        
        # Apply physical constraints
        self._apply_constraints()
        
        # Store current field in history
        self.history.append(self.field.copy())
        
        # Update metrics
        self._update_metrics()

    def _update_base_field(self, phase_field) -> None:
        """
        Update the base energy field from phase field.

        Args:
            phase_field: Current phase field
        """
        # Extract phase and amplitude from phase field
        amplitude = np.abs(phase_field.field)
        phase_gradient = phase_field.calculate_phase_gradient()
        
        # Base energy proportional to amplitude squared and phase gradient
        gradient_energy = np.sum(phase_gradient**2, axis=0)
        
        # Calculate phase singularity contribution
        singularity_energy = np.zeros(self.grid_size)
        for singularity in phase_field.singularities:
            pos = singularity['position']
            charge = singularity['charge']
            
            # Create energy peak around singularity with sign based on charge
            for i in range(max(0, pos[0]-3), min(self.grid_size[0], pos[0]+4)):
                for j in range(max(0, pos[1]-3), min(self.grid_size[1], pos[1]+4)):
                    for k in range(max(0, pos[2]-3), min(self.grid_size[2], pos[2]+4)):
                        dist = np.sqrt((i-pos[0])**2 + (j-pos[1])**2 + (k-pos[2])**2)
                        if dist < 3:
                            # Energy peak with sign based on charge
                            singularity_energy[i, j, k] += charge * 2.0 * np.exp(-dist**2 / 1.0)
        
        # Combine components with weights
        self.base_field = (
            0.3 * amplitude**2 + 
            0.4 * gradient_energy + 
            0.3 * singularity_energy
        )
        
        # Normalize base field
        max_val = np.max(self.base_field)
        if max_val > 0:
            self.base_field = self.base_field / max_val

    def _add_feedback(self, phase_field) -> None:
        """Add non-local feedback contribution to the energy field."""
        # Initialize with base field
        new_field = self.base_field.copy()
        
        # Skip feedback if not enough history
        if len(self.history) < 2:
            self.field = new_field
            return
        
        # Get grid coordinates
        nx, ny, nz = self.grid_size
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(-1, 1, nz)
        
        # For efficiency, use a sparse sampling of points for feedback
        # Select every 3rd point in each dimension
        sample_points = []
        for i in range(0, nx, 3):
            for j in range(0, ny, 3):
                for k in range(0, nz, 3):
                    sample_points.append((i, j, k))
        
        # Pre-compute phase field values
        phase_values = np.angle(phase_field.field)
        
        # Apply non-local feedback for each point
        for i, j, k in np.ndindex(nx, ny, nz):
            feedback = 0.0
            current_pos = np.array([x[i], y[j], z[k]])
            
            # Add contributions from sampled points
            for sp_i, sp_j, sp_k in sample_points:
                # Skip if same point
                if i == sp_i and j == sp_j and k == sp_k:
                    continue
                
                # Calculate distance
                source_pos = np.array([x[sp_i], y[sp_j], z[sp_k]])
                distance = np.linalg.norm(current_pos - source_pos)
                
                if distance < 1e-10:
                    continue
                
                # Calculate delay based on distance
                delay = int(distance / 0.1)
                delay = min(delay, len(self.history) - 1)
                
                # Get delayed energy value
                delayed_energy = self.history[-delay-1][sp_i, sp_j, sp_k]
                
                # Calculate kernel value with distance decay
                kernel = self.kappa / (distance**self.alpha)
                
                # Add phase coupling
                phase_diff = phase_values[i, j, k] - phase_values[sp_i, sp_j, sp_k]
                phase_factor = np.cos(phase_diff)
                kernel *= (1.0 + self.phi_coupling * phase_factor)
                
                # Add contribution
                feedback += kernel * delayed_energy
            
            # Add feedback to base field (with nonlinear saturation)
            saturation_factor = 1.0 / (1.0 + np.exp(new_field[i, j, k] - 0.8))
            new_field[i, j, k] += feedback * saturation_factor
        
        # Update field
        self.field = new_field

    def _apply_constraints(self) -> None:
        """Apply physical constraints to the energy field."""
        # Ensure positive energy values
        self.field = np.maximum(0, self.field)
        
        # Apply upper bound to prevent unbounded growth
        self.field = np.minimum(self.field, 2.0)
        
        # Apply conservation of energy (optional)
        # If enabled, this would ensure total energy remains constant
        if False:  # Disabled by default
            target_energy = np.sum(self.base_field)
            current_energy = np.sum(self.field)
            
            if current_energy > 0:
                self.field = self.field * (target_energy / current_energy)

    def _update_metrics(self) -> None:
        """Update energy field metrics."""
        # Calculate total energy
        self.total_energy = np.sum(self.field)
        
        # Find peak energy
        self.peak_energy = np.max(self.field)
        
        # Calculate energy gradient magnitude
        gradient = self.calculate_energy_gradient()
        self.energy_gradient_magnitude = np.mean(gradient)

    def calculate_energy_gradient(self) -> float:
        """
        Calculate the average magnitude of the energy gradient.
        
        Returns:
            Average gradient magnitude
        """
        nx, ny, nz = self.grid_size
        gradient_magnitude = np.zeros(self.grid_size)
        
        # Calculate gradients using central differences for interior points
        # X-direction
        gradient_x = np.zeros((nx, ny, nz))
        gradient_x[1:-1, :, :] = (self.field[2:, :, :] - self.field[:-2, :, :]) / (2 * self.dx)
        
        # Y-direction
        gradient_y = np.zeros((nx, ny, nz))
        gradient_y[:, 1:-1, :] = (self.field[:, 2:, :] - self.field[:, :-2, :]) / (2 * self.dx)
        
        # Z-direction
        gradient_z = np.zeros((nx, ny, nz))
        gradient_z[:, :, 1:-1] = (self.field[:, :, 2:] - self.field[:, :, :-2]) / (2 * self.dx)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)
        
        return np.mean(gradient_magnitude)

    def get_peak_energy(self) -> float:
        """
        Get the peak energy value in the field.
        
        Returns:
            Peak energy value
        """
        return self.peak_energy

    def calculate_total_energy(self) -> float:
        """
        Calculate the total energy in the field.
        
        Returns:
            Total energy value
        """
        return self.total_energy

    def detect_energy_centers(self, threshold: float = 0.7) -> List[Dict]:
        """
        Detect energy centers in the field.
        
        Args:
            threshold: Energy threshold for detection (0.0-1.0)
            
        Returns:
            List of energy centers with positions and values
        """
        centers = []
        nx, ny, nz = self.grid_size
        
        # Normalize threshold to peak energy
        absolute_threshold = threshold * self.peak_energy
        
        # Find local maxima above threshold
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    value = self.field[i, j, k]
                    
                    if value < absolute_threshold:
                        continue
                        
                    # Check if local maximum
                    is_maximum = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                    
                                if i+di < 0 or i+di >= nx or j+dj < 0 or j+dj >= ny or k+dk < 0 or k+dk >= nz:
                                    continue
                                    
                                if self.field[i+di, j+dj, k+dk] > value:
                                    is_maximum = False
                                    break
                            if not is_maximum:
                                break
                        if not is_maximum:
                            break
                            
                    if is_maximum:
                        # Convert indices to spatial coordinates
                        x = -1 + (2 * i / (nx - 1)) if nx > 1 else 0
                        y = -1 + (2 * j / (ny - 1)) if ny > 1 else 0
                        z = -1 + (2 * k / (nz - 1)) if nz > 1 else 0
                        
                        centers.append({
                            "position": (i, j, k),
                            "coordinates": (x, y, z),
                            "value": float(value),
                            "relative_value": float(value / self.peak_energy)
                        })
        
        # Sort by energy value (descending)
        centers.sort(key=lambda c: c["value"], reverse=True)
        
        return centers

    def find_resonant_channels(self) -> List[Dict]:
        """
        Find resonant energy channels between energy centers.
        
        Returns:
            List of resonant channels with source, target, and strength
        """
        # First detect energy centers
        centers = self.detect_energy_centers(threshold=0.5)
        
        # Find channels between centers
        channels = []
        
        for i, center1 in enumerate(centers):
            pos1 = center1["position"]
            
            for j, center2 in enumerate(centers[i+1:], i+1):
                pos2 = center2["position"]
                
                # Calculate distance
                distance = np.sqrt(
                    (pos1[0] - pos2[0])**2 + 
                    (pos1[1] - pos2[1])**2 + 
                    (pos1[2] - pos2[2])**2
                )
                
                # Skip if too far apart
                if distance > 10:
                    continue
                
                # Check for energy channel between centers
                channel_strength = self._measure_channel_strength(pos1, pos2)
                
                # If channel is strong enough, add to list
                if channel_strength > 0.3:
                    channels.append({
                        "source": i,
                        "target": j,
                        "source_position": pos1,
                        "target_position": pos2,
                        "distance": distance,
                        "strength": channel_strength
                    })
        
        return channels

    def _measure_channel_strength(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """
        Measure the strength of an energy channel between two positions.
        
        Args:
            pos1, pos2: Grid positions to check
            
        Returns:
            Channel strength (0.0-1.0)
        """
        # Calculate distance
        distance = np.sqrt(
            (pos1[0] - pos2[0])**2 + 
            (pos1[1] - pos2[1])**2 + 
            (pos1[2] - pos2[2])**2
        )
        
        # Calculate discrete line points between pos1 and pos2
        num_points = int(distance) + 1
        line_points = []
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            x = int(pos1[0] + t * (pos2[0] - pos1[0]))
            y = int(pos1[1] + t * (pos2[1] - pos1[1]))
            z = int(pos1[2] + t * (pos2[2] - pos1[2]))
            line_points.append((x, y, z))
        
        # Measure energy along the line
        energy_values = [self.field[p] for p in line_points]
        
        # Calculate channel strength metrics
        min_energy = min(energy_values)
        mean_energy = np.mean(energy_values)
        
        # Calculate energy gradient uniformity
        gradients = np.diff(energy_values)
        gradient_uniformity = 1.0 - np.std(gradients) / (np.mean(np.abs(gradients)) + 1e-10)
        
        # Combined strength metric
        source_energy = self.field[pos1]
        target_energy = self.field[pos2]
        endpoint_ratio = min(source_energy, target_energy) / max(source_energy, target_energy)
        
        # Energy should remain relatively high along the channel
        min_ratio = min_energy / max(source_energy, target_energy)
        
        # Combine metrics
        strength = 0.4 * min_ratio + 0.3 * endpoint_ratio + 0.3 * gradient_uniformity
        
        return strength

    def apply_phase_coupling(self, phase_field, coupling_strength: float = 0.2) -> None:
        """
        Apply coupling between energy and phase fields.
        
        Args:
            phase_field: Phase field object
            coupling_strength: Strength of coupling (0.0-1.0)
        """
        # Get phase field data
        phase_values = np.angle(phase_field.field)
        phase_amplitude = np.abs(phase_field.field)
        
        # Calculate phase gradient
        phase_gradient = phase_field.calculate_phase_gradient()
        gradient_magnitude = np.sqrt(np.sum(phase_gradient**2, axis=0))
        
        # Apply coupling
        # Energy affects phase field amplitude (implemented in phase_field.update)
        # Phase affects energy field distribution
        energy_modulation = 1.0 + coupling_strength * np.sin(phase_values)
        
        # Apply modulation to energy field
        modulated_field = self.field * energy_modulation
        
        # Add energy concentration at high phase gradient locations
        gradient_contribution = coupling_strength * gradient_magnitude / (np.max(gradient_magnitude) + 1e-10)
        modulated_field += gradient_contribution
        
        # Apply constraints
        modulated_field = np.maximum(0, modulated_field)
        
        # Update field
        self.field = modulated_field
        
        # Update metrics
        self._update_metrics()

    def get_serializable_state(self) -> Dict:
        """
        Get a serializable representation of the energy field.

        Returns:
            Dictionary with field state
        """
        return {
            'field': self.field.copy(),
            'base_field': self.base_field.copy(),
            'history': [h.copy() for h in self.history],
            'total_energy': self.total_energy,
            'peak_energy': self.peak_energy,
            'energy_gradient_magnitude': self.energy_gradient_magnitude
        }
    
    def load_state(self, state: Dict) -> None:
        """
        Load field state from a dictionary.

        Args:
            state: Dictionary with field state
        """
        self.field = state['field']
        self.base_field = state['base_field']
        self.history = deque(state['history'], maxlen=self.history_length)
        self.total_energy = state['total_energy']
        self.peak_energy = state['peak_energy']
        self.energy_gradient_magnitude = state['energy_gradient_magnitude']
                