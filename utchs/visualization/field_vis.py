"""
Field visualization module for the UTCHS framework.

This module implements visualization tools for the phase and energy fields,
including 2D slices, 3D visualizations, and animations of field evolution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

class FieldVisualizer:
    """
    Visualizes phase and energy fields using matplotlib.
    """
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Initialize the field visualizer.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up figure style
        plt.style.use('dark_background')
        self.cmap = 'viridis'
        self.phase_cmap = 'hsv'
        self.dpi = 300
        
        # Custom colormaps
        self.setup_custom_colormaps()
        
    def setup_custom_colormaps(self):
        """Set up custom colormaps for field visualization."""
        # Define a custom colormap for phase that loops smoothly
        phase_colors = plt.cm.hsv(np.linspace(0, 1, 256))
        self.phase_cmap_custom = LinearSegmentedColormap.from_list('phase_cmap', phase_colors)
        
        # Define a custom colormap for energy
        energy_colors = [
            (0, 0, 0.5),        # Dark blue for low energy
            (0, 0, 1),          # Blue
            (0, 1, 1),          # Cyan
            (0, 1, 0),          # Green
            (1, 1, 0),          # Yellow
            (1, 0.5, 0),        # Orange
            (1, 0, 0)           # Red for high energy
        ]
        self.energy_cmap_custom = LinearSegmentedColormap.from_list('energy_cmap', energy_colors)

    def visualize_phase_slice(
        self,
        phase_field,
        axis: int = 2,
        slice_idx: Optional[int] = None,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize a 2D slice of the phase field.

        Args:
            phase_field: PhaseField instance
            axis: Axis for the slice (0=x, 1=y, 2=z)
            slice_idx: Index for the slice (default: middle)
            save: Whether to save the figure
            filename: Custom filename (default: auto-generated)

        Returns:
            matplotlib Figure
        """
        field = phase_field.field
        grid_size = field.shape
        
        # Default to middle slice
        if slice_idx is None:
            slice_idx = grid_size[axis] // 2
            
        # Extract 2D slice
        if axis == 0:
            slice_data = field[slice_idx, :, :]
            xlabel, ylabel = 'Y', 'Z'
        elif axis == 1:
            slice_data = field[:, slice_idx, :]
            xlabel, ylabel = 'X', 'Z'
        else:  # axis == 2
            slice_data = field[:, :, slice_idx]
            xlabel, ylabel = 'X', 'Y'
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot phase
        phase = np.angle(slice_data)
        im1 = ax1.imshow(phase.T, origin='lower', cmap=self.phase_cmap, vmin=-np.pi, vmax=np.pi)
        ax1.set_title(f'Phase (slice {slice_idx} along {["X", "Y", "Z"][axis]} axis)')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        fig.colorbar(im1, ax=ax1, label='Phase [rad]')
        
        # Plot amplitude
        amplitude = np.abs(slice_data)
        im2 = ax2.imshow(amplitude.T, origin='lower', cmap=self.cmap)
        ax2.set_title(f'Amplitude (slice {slice_idx} along {["X", "Y", "Z"][axis]} axis)')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        fig.colorbar(im2, ax=ax2, label='Amplitude')
        
        # Add singularities
        for singularity in phase_field.singularities:
            position = singularity['position']
            charge = singularity['charge']
            
            # Check if singularity is in this slice
            if (axis == 0 and position[0] == slice_idx) or \
               (axis == 1 and position[1] == slice_idx) or \
               (axis == 2 and position[2] == slice_idx):
                
                # Get 2D coordinates
                if axis == 0:
                    x, y = position[1], position[2]
                elif axis == 1:
                    x, y = position[0], position[2]
                else:  # axis == 2
                    x, y = position[0], position[1]
                    
                # Plot marker
                color = 'red' if charge > 0 else 'blue'
                ax1.plot(x, y, 'o', color=color, markersize=8)
                ax2.plot(x, y, 'o', color=color, markersize=8)
                
        plt.tight_layout()
        
        # Save figure
        if save:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'phase_slice_{["x", "y", "z"][axis]}{slice_idx}_{timestamp}.png'
                
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved figure to {filepath}")
            
        return fig

    def visualize_energy_slice(
        self,
        energy_field,
        axis: int = 2,
        slice_idx: Optional[int] = None,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize a 2D slice of the energy field.

        Args:
            energy_field: EnergyField instance
            axis: Axis for the slice (0=x, 1=y, 2=z)
            slice_idx: Index for the slice (default: middle)
            save: Whether to save the figure
            filename: Custom filename (default: auto-generated)

        Returns:
            matplotlib Figure
        """
        field = energy_field.field
        base_field = energy_field.base_field
        grid_size = field.shape
        
        # Default to middle slice
        if slice_idx is None:
            slice_idx = grid_size[axis] // 2
            
        # Extract 2D slice
        if axis == 0:
            slice_data = field[slice_idx, :, :]
            base_slice = base_field[slice_idx, :, :]
            xlabel, ylabel = 'Y', 'Z'
        elif axis == 1:
            slice_data = field[:, slice_idx, :]
            base_slice = base_field[:, slice_idx, :]
            xlabel, ylabel = 'X', 'Z'
        else:  # axis == 2
            slice_data = field[:, :, slice_idx]
            base_slice = base_field[:, :, slice_idx]
            xlabel, ylabel = 'X', 'Y'
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot total energy
        im1 = ax1.imshow(slice_data.T, origin='lower', cmap=self.energy_cmap_custom)
        ax1.set_title(f'Total Energy (slice {slice_idx} along {["X", "Y", "Z"][axis]})')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        
        # Plot base energy
        im2 = ax2.imshow(base_slice.T, origin='lower', cmap=self.cmap)
        ax2.set_title(f'Base Energy (slice {slice_idx} along {["X", "Y", "Z"][axis]} axis)')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        fig.colorbar(im1, ax=ax1, label='Total Energy')
        fig.colorbar(im2, ax=ax2, label='Base Energy')
        
        # Add singularities
        for singularity in energy_field.singularities:
            position = singularity['position']
            charge = singularity['charge']
            
            # Check if singularity is in this slice
            if (axis == 0 and position[0] == slice_idx) or \
               (axis == 1 and position[1] == slice_idx) or \
               (axis == 2 and position[2] == slice_idx):
                
                # Get 2D coordinates
                if axis == 0:
                    x, y = position[1], position[2]
                elif axis == 1:
                    x, y = position[0], position[2]
                else:  # axis == 2
                    x, y = position[0], position[1]
                    
                # Plot marker
                color = 'red' if charge > 0 else 'blue'
                ax1.plot(x, y, 'o', color=color, markersize=8)
                ax2.plot(x, y, 'o', color=color, markersize=8)
                
        plt.tight_layout()
        
        # Save figure
        if save:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'energy_slice_{["x", "y", "z"][axis]}{slice_idx}_{timestamp}.png'
                
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved figure to {filepath}")
            
        return fig