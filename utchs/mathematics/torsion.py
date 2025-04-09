"""
Torsion field module for the UTCHS framework.

This module implements the TorsionField class, which calculates and analyzes
the torsion tensor that represents how the phase field "twists" around each point.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class TorsionField:
    """
    Implements the torsion field tensor calculation and analysis.

    The torsion tensor is defined as:
    T_{ijk}(z) = ∂_i φ_j(z) - ∂_j φ_i(z) + ω_{ijk}(z)
    """
    def __init__(self, grid_size: Tuple[int, int, int], dx: float = 0.1):
        """
        Initialize the torsion field.

        Args:
            grid_size: Tuple of (nx, ny, nz) grid dimensions
            dx: Grid spacing
        """
        self.grid_size = grid_size
        self.dx = dx
        
        # Initialize phase field components (3D vector field)
        self.phi = np.zeros((3, *grid_size), dtype=float)
        
        # Initialize connection coefficients
        self.omega = np.zeros((3, 3, 3, *grid_size), dtype=float)
        
        # Initialize torsion tensor
        self.tensor = np.zeros((3, 3, 3, *grid_size), dtype=float)
        
        # Initialize field metric for stability analysis
        self.stability_metric = 0.0
        
        # Store lattice points
        self.lattice_points = []

    def update_phase_field(self, phi: np.ndarray) -> None:
        """
        Update the phase field components.

        Args:
            phi: New phase field values (3D vector field)
        """
        if phi.shape != self.phi.shape:
            raise ValueError(f"Shape mismatch: expected {self.phi.shape}, got {phi.shape}")
            
        self.phi = phi.copy()
        self._update_torsion_tensor()
        self._update_stability_metric()
        self._find_lattice_points()

    def update_connection(self, omega: np.ndarray) -> None:
        """
        Update the connection coefficients.

        Args:
            omega: New connection coefficients
        """
        if omega.shape != self.omega.shape:
            raise ValueError(f"Shape mismatch: expected {self.omega.shape}, got {omega.shape}")
            
        self.omega = omega.copy()
        self._update_torsion_tensor()
        self._update_stability_metric()
        self._find_lattice_points()

    def _update_connection_coefficients(self) -> None:
        """
        Calculate connection coefficients based on the phase field geometry.
        
        This implements a physical model where the connection coefficients
        depend on the local phase field structure.
        """
        nx, ny, nz = self.grid_size
        
        # Calculate the first derivatives of the phase field
        # This embeds the recursive structure of the field into the connection coefficients
        
        # Get phase field gradients
        phi_grad = self._calculate_gradients(self.phi)
        
        # Compute the connection coefficients based on field structure
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Nonlinear coupling term based on field gradients
                    self.omega[i, j, k] = 0.1 * (
                        phi_grad[i] * phi_grad[j] * phi_grad[k]
                    )
                    
                    # Add recursive term based on the tensor itself
                    # (This creates the self-referential structure)
                    if hasattr(self, 'tensor') and np.any(self.tensor):
                        self.omega[i, j, k] += 0.05 * np.sin(self.tensor[i, j, k])

    def _calculate_gradients(self, field: np.ndarray) -> np.ndarray:
        """
        Calculate gradients of a field using finite differences.
        
        Args:
            field: Input field to differentiate (shape: (C, nx, ny, nz))
            
        Returns:
            Gradient field (shape: (C, 3, nx, ny, nz))
        """
        C = field.shape[0]  # Number of components
        nx, ny, nz = self.grid_size
        gradient = np.zeros((C, 3, nx, ny, nz), dtype=float)
        
        # For each component
        for c in range(C):
            # x-derivative (component 0)
            gradient[c, 0, 1:-1, :, :] = (field[c, 2:, :, :] - field[c, :-2, :, :]) / (2 * self.dx)
            gradient[c, 0, 0, :, :] = (field[c, 1, :, :] - field[c, 0, :, :]) / self.dx
            gradient[c, 0, -1, :, :] = (field[c, -1, :, :] - field[c, -2, :, :]) / self.dx
            
            # y-derivative (component 1)
            gradient[c, 1, :, 1:-1, :] = (field[c, :, 2:, :] - field[c, :, :-2, :]) / (2 * self.dx)
            gradient[c, 1, :, 0, :] = (field[c, :, 1, :] - field[c, :, 0, :]) / self.dx
            gradient[c, 1, :, -1, :] = (field[c, :, -1, :] - field[c, :, -2, :]) / self.dx
            
            # z-derivative (component 2)
            gradient[c, 2, :, :, 1:-1] = (field[c, :, :, 2:] - field[c, :, :, :-2]) / (2 * self.dx)
            gradient[c, 2, :, :, 0] = (field[c, :, :, 1] - field[c, :, :, 0]) / self.dx
            gradient[c, 2, :, :, -1] = (field[c, :, :, -1] - field[c, :, :, -2]) / self.dx
            
        return gradient

    def _update_torsion_tensor(self) -> None:
        """Calculate the torsion tensor from phase field and connection."""
        nx, ny, nz = self.grid_size
        
        # Calculate gradients for all components
        phi_grad = np.zeros((3, 3, nx, ny, nz), dtype=float)
        
        for i in range(3):
            for j in range(3):
                # Calculate ∂_i φ_j using finite differences
                if i == 0:  # x-derivative
                    # Interior points: central difference
                    phi_grad[i, j, 1:-1, :, :] = (self.phi[j, 2:, :, :] - self.phi[j, :-2, :, :]) / (2 * self.dx)
                    # Boundaries: forward/backward difference
                    phi_grad[i, j, 0, :, :] = (self.phi[j, 1, :, :] - self.phi[j, 0, :, :]) / self.dx
                    phi_grad[i, j, -1, :, :] = (self.phi[j, -1, :, :] - self.phi[j, -2, :, :]) / self.dx
                    
                elif i == 1:  # y-derivative
                    # Interior points: central difference
                    phi_grad[i, j, :, 1:-1, :] = (self.phi[j, :, 2:, :] - self.phi[j, :, :-2, :]) / (2 * self.dx)
                    # Boundaries: forward/backward difference
                    phi_grad[i, j, :, 0, :] = (self.phi[j, :, 1, :] - self.phi[j, :, 0, :]) / self.dx
                    phi_grad[i, j, :, -1, :] = (self.phi[j, :, -1, :] - self.phi[j, :, -2, :]) / self.dx
                    
                elif i == 2:  # z-derivative
                    # Interior points: central difference
                    phi_grad[i, j, :, :, 1:-1] = (self.phi[j, :, :, 2:] - self.phi[j, :, :, :-2]) / (2 * self.dx)
                    # Boundaries: forward/backward difference
                    phi_grad[i, j, :, :, 0] = (self.phi[j, :, :, 1] - self.phi[j, :, :, 0]) / self.dx
                    phi_grad[i, j, :, :, -1] = (self.phi[j, :, :, -1] - self.phi[j, :, :, -2]) / self.dx
        
        # Update connection coefficients based on current field structure
        self._update_connection_coefficients()
        
        # Calculate tensor components: T_{ijk} = ∂_i φ_j - ∂_j φ_i + ω_{ijk}
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.tensor[i, j, k] = phi_grad[i, j] - phi_grad[j, i] + self.omega[i, j, k]

    def _update_stability_metric(self) -> None:
        """Calculate the stability metric for the torsion field."""
        # Calculate curl of torsion tensor
        curl_norm_squared = self._calculate_curl_norm_squared()
        
        # Calculate tensor norm squared
        tensor_norm_squared = np.sum(self.tensor**2)
        
        # Stability condition: ∫ ||∇ × T(z)||² dz ≤ ∫ ||T(z)||² dz
        self.stability_metric = tensor_norm_squared - curl_norm_squared

    def _calculate_curl_norm_squared(self) -> float:
        """Calculate the squared norm of the curl of the torsion tensor."""
        # Simplified calculation of the curl of the tensor
        nx, ny, nz = self.grid_size
        curl = np.zeros_like(self.tensor)
        
        # Calculate curl components using finite differences
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # x-component of curl
                    if j < 2 and k < 2:  # Avoid index errors
                        curl[i, j, k, 1:-1, :, :] += (
                            (self.tensor[i, j+1, k, 2:, :, :] - self.tensor[i, j+1, k, :-2, :, :]) / (2 * self.dx) -
                            (self.tensor[i, j, k+1, 2:, :, :] - self.tensor[i, j, k+1, :-2, :, :]) / (2 * self.dx)
                        )
                    
                    # y-component of curl
                    if i < 2 and k < 2:  # Avoid index errors
                        curl[i, j, k, :, 1:-1, :] += (
                            (self.tensor[i+1, j, k, :, 2:, :] - self.tensor[i+1, j, k, :, :-2, :]) / (2 * self.dx) -
                            (self.tensor[i, j, k+1, :, 2:, :] - self.tensor[i, j, k+1, :, :-2, :]) / (2 * self.dx)
                        )
                    
                    # z-component of curl
                    if i < 2 and j < 2:  # Avoid index errors
                        curl[i, j, k, :, :, 1:-1] += (
                            (self.tensor[i+1, j, k, :, :, 2:] - self.tensor[i+1, j, k, :, :, :-2]) / (2 * self.dx) -
                            (self.tensor[i, j+1, k, :, :, 2:] - self.tensor[i, j+1, k, :, :, :-2]) / (2 * self.dx)
                        )
        
        # Calculate norm squared of curl
        curl_norm_squared = np.sum(curl**2)
        return curl_norm_squared

    def _find_lattice_points(self, threshold: float = 0.1) -> None:
        """
        Find lattice points where the torsion field exhibits specific patterns.
        
        Lattice points are defined as positions where the torsion field has
        characteristic signatures, indicating stable configurations.
        
        Args:
            threshold: Threshold value for pattern detection
        """
        nx, ny, nz = self.grid_size
        lattice_points = []
        
        # Define pattern operator (simplified version)
        # Look for points where the torsion field has stabilized
        def pattern_operator(tensor_at_point):
            # Calculate tensor invariants
            # First invariant: trace-like sum
            invariant1 = np.sum([tensor_at_point[i, i, i] for i in range(3)])
            
            # Second invariant: determinant-like product
            invariant2 = np.sum([
                tensor_at_point[i, j, k] * tensor_at_point[j, k, i] * tensor_at_point[k, i, j]
                for i in range(3) for j in range(3) for k in range(3)
            ])
            
            # Pattern condition: close to specific values
            return abs(invariant1) < threshold and abs(invariant2 - 1.0) < threshold
        
        # Scan the field for lattice points
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    tensor_at_point = self.tensor[:, :, :, i, j, k]
                    
                    if pattern_operator(tensor_at_point):
                        # Found a lattice point
                        lattice_points.append({
                            'position': (i, j, k),
                            'stability': self._calculate_local_stability(i, j, k)
                        })
        
        self.lattice_points = lattice_points

    def _calculate_local_stability(self, i: int, j: int, k: int) -> float:
        """
        Calculate the local stability at a specific point.
        
        Args:
            i, j, k: Grid indices
            
        Returns:
            Stability value
        """
        # Extract local tensor
        local_tensor = self.tensor[:, :, :, i, j, k]
        
        # Eigenvalue-based stability calculation
        # Create stability matrix from tensor
        stability_matrix = np.zeros((9, 9))
        
        for i1 in range(3):
            for j1 in range(3):
                for i2 in range(3):
                    for j2 in range(3):
                        row = i1*3 + j1
                        col = i2*3 + j2
                        stability_matrix[row, col] = local_tensor[i1, j1, i2]
        
        # Calculate eigenvalues
        try:
            eigenvalues = np.linalg.eigvals(stability_matrix)
            # Stability indicated by negative real parts of eigenvalues
            return -np.sum(np.real(eigenvalues))
        except np.linalg.LinAlgError:
            # In case of numerical issues
            return 0.0
    
    def calculate_stability(self) -> float:
        """
        Calculate the overall stability metric for the torsion field.
        
        Returns:
            Stability value (higher is more stable)
        """
        return self.stability_metric
    
    def get_lattice_points(self) -> List[Dict]:
        """
        Get the detected lattice points.
        
        Returns:
            List of lattice points with their positions and stability values
        """
        return self.lattice_points
    
    def get_serializable_state(self) -> Dict:
        """
        Get a serializable representation of the torsion field.
        
        Returns:
            Dictionary with field state
        """
        return {
            'phi': self.phi,
            'tensor': self.tensor,
            'stability_metric': self.stability_metric,
            'lattice_points': self.lattice_points
        }
    
    def load_state(self, state: Dict) -> None:
        """
        Load torsion field state from a dictionary.
        
        Args:
            state: Dictionary with field state
        """
        self.phi = state['phi']
        self.tensor = state['tensor']
        self.stability_metric = state['stability_metric']
        self.lattice_points = state['lattice_points']
        
        # Recalculate connection coefficients
        self._update_connection_coefficients()