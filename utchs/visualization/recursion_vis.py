"""
Recursion visualization module for the UTCHS framework.

This module implements the RecursionVisualizer class, which provides visualization
tools for analyzing relationships across recursive levels in the UTCHS framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict

from ..utils.logging_config import get_logger
from ..core.recursion_tracker import RecursionTracker
from ..core.transition_analyzer import TransitionAnalyzer
from ..core.fractal_analyzer import FractalAnalyzer

logger = get_logger(__name__)

class RecursionVisualizer:
    """
    Visualizes relationships across recursive levels.
    
    This class provides tools for visualizing how positions and patterns
    evolve across different recursive levels (octaves) in the UTCHS system.
    """
    
    def __init__(self, recursion_tracker: RecursionTracker, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the recursion visualizer.
        
        Args:
            recursion_tracker: RecursionTracker instance containing position history
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.recursion_tracker = recursion_tracker
        
        # Common figure size and DPI settings
        self.fig_size = self.config.get('fig_size', (12, 8))
        self.dpi = self.config.get('dpi', 100)
        
        # Color maps
        self.depth_cmap = plt.cm.viridis
        self.position_cmap = plt.cm.hsv
        
        # Golden ratio (φ) for reference
        self.phi = (1 + np.sqrt(5)) / 2
        
        logger.info("RecursionVisualizer initialized")
    
    def visualize_position_across_depths(self, position_number: int, tick: int, 
                                         save: bool = False, filename: Optional[str] = None) -> plt.Figure:
        """
        Visualize a specific position across all recursion depths.
        
        Args:
            position_number: Position number (1-13)
            tick: Current system tick
            save: Whether to save the figure
            filename: Output filename (if save is True)
            
        Returns:
            Matplotlib figure
        """
        # Get position data across all depths
        position_data = self.recursion_tracker.get_position_across_recursion_levels(position_number, tick)
        
        if not position_data:
            logger.warning(f"No data available for position {position_number} at tick {tick}")
            fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
            ax.text(0.5, 0.5, f"No data available for position {position_number}", 
                   ha='center', va='center', fontsize=14)
            plt.close()
            return fig
        
        # Create figure
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        # Create 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Extract data for plotting
        depths = sorted(position_data.keys())
        x = []
        y = []
        z = []
        energy = []
        phase = []
        
        for depth in depths:
            data = position_data[depth]
            coords = data['spatial_location']
            x.append(coords[0])
            y.append(coords[1])
            z.append(coords[2])
            energy.append(data['energy_level'])
            phase.append(data['phase'])
        
        # Normalize energy for size
        min_size = 30
        max_size = 200
        if energy:
            min_energy = min(energy)
            max_energy = max(energy)
            if min_energy != max_energy:
                normalized_sizes = [min_size + (e - min_energy) * (max_size - min_size) / (max_energy - min_energy) for e in energy]
            else:
                normalized_sizes = [min_size] * len(energy)
        else:
            normalized_sizes = [min_size] * len(depths)
        
        # Plot points
        for i, depth in enumerate(depths):
            ax1.scatter(x[i], y[i], z[i], 
                       s=normalized_sizes[i], 
                       c=[self.depth_cmap(depth / self.recursion_tracker.max_recursion_depth)],
                       alpha=0.7, 
                       edgecolor='white', 
                       label=f"Depth {depth}")
        
        # Add connections between consecutive depths
        for i in range(len(depths) - 1):
            ax1.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], 
                    color='gray', linestyle='--', alpha=0.5)
        
        ax1.set_xlabel('X', fontsize=10)
        ax1.set_ylabel('Y', fontsize=10)
        ax1.set_zlabel('Z', fontsize=10)
        ax1.set_title(f'Position {position_number} Across Recursion Levels (3D)', fontsize=12)
        
        # Create 2D plot for energy and phase
        ax2 = fig.add_subplot(122)
        
        # Plot energy levels across depths
        ax2.plot(depths, energy, 'o-', color='blue', label='Energy')
        ax2.set_xlabel('Recursion Depth', fontsize=10)
        ax2.set_ylabel('Energy Level', fontsize=10, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Create second y-axis for phase
        ax3 = ax2.twinx()
        ax3.plot(depths, phase, 'o-', color='red', label='Phase')
        ax3.set_ylabel('Phase', fontsize=10, color='red')
        ax3.tick_params(axis='y', labelcolor='red')
        
        # Add phi reference line for energy scaling
        if len(depths) > 1:
            # Calculate phi-scaled energy values
            phi_energy = [energy[0]]
            for i in range(1, len(depths)):
                phi_energy.append(phi_energy[i-1] * self.phi)
            
            ax2.plot(depths, phi_energy, '--', color='green', alpha=0.5, label='φ-Scaled')
        
        # Create combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax2.set_title(f'Energy and Phase Across Recursion Levels', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            if filename is None:
                filename = f"position_{position_number}_recursion_vis_tick_{tick}.png"
            plt.savefig(filename, dpi=self.dpi)
            logger.info(f"Saved visualization to {filename}")
        
        return fig
    
    def visualize_phi_scaling(self, analyzer: FractalAnalyzer, 
                               save: bool = False, filename: Optional[str] = None) -> plt.Figure:
        """
        Visualize phi (golden ratio) scaling across recursion depths.
        
        Args:
            analyzer: FractalAnalyzer instance with scaling metrics
            save: Whether to save the figure
            filename: Output filename (if save is True)
            
        Returns:
            Matplotlib figure
        """
        # Get scaling invariance data
        scaling_data = analyzer.fractal_metrics.get('scaling_invariance')
        
        if not scaling_data or 'position_scaling_factors' not in scaling_data:
            logger.warning("No scaling invariance data available")
            fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
            ax.text(0.5, 0.5, "No scaling invariance data available", 
                   ha='center', va='center', fontsize=14)
            plt.close()
            return fig
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.fig_size, dpi=self.dpi)
        
        # Extract data
        position_scaling = scaling_data['position_scaling_factors']
        positions = sorted(position_scaling.keys())
        
        # Calculate metrics for plotting
        mean_factors = [position_scaling[p]['mean_factor'] for p in positions]
        std_factors = [position_scaling[p]['std_factor'] for p in positions]
        
        # Plot mean scaling factors by position
        ax1.bar(positions, mean_factors, yerr=std_factors, 
               color=[self.position_cmap(p/13) for p in positions],
               alpha=0.7, edgecolor='black', capsize=5)
        
        # Add phi reference line
        ax1.axhline(y=self.phi, color='red', linestyle='--', alpha=0.7, label=f'φ = {self.phi:.3f}')
        
        ax1.set_xlabel('Position Number', fontsize=10)
        ax1.set_ylabel('Mean Scaling Factor', fontsize=10)
        ax1.set_xticks(positions)
        ax1.set_title('Scaling Factors by Position', fontsize=12)
        ax1.legend()
        
        # Highlight invariant positions
        invariant_positions = scaling_data.get('invariant_positions', [])
        if invariant_positions:
            for pos in invariant_positions:
                ax1.bar(pos, mean_factors[positions.index(pos)], 
                       color='none', edgecolor='green', linewidth=2)
        
        # Create histogram of all scaling factors
        all_factors = []
        for p in positions:
            all_factors.extend(position_scaling[p]['factors'])
            
        if all_factors:
            ax2.hist(all_factors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add phi reference line
            ax2.axvline(x=self.phi, color='red', linestyle='--', alpha=0.7, label=f'φ = {self.phi:.3f}')
            
            # Add mean reference line
            mean_factor = np.mean(all_factors)
            ax2.axvline(x=mean_factor, color='green', linestyle='-', alpha=0.7, 
                       label=f'Mean = {mean_factor:.3f}')
            
            ax2.set_xlabel('Scaling Factor', fontsize=10)
            ax2.set_ylabel('Frequency', fontsize=10)
            ax2.set_title('Distribution of Scaling Factors', fontsize=12)
            ax2.legend()
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            if filename is None:
                filename = "phi_scaling_visualization.png"
            plt.savefig(filename, dpi=self.dpi)
            logger.info(f"Saved visualization to {filename}")
        
        return fig
    
    def visualize_p13_seventh_cycle(self, analyzer: TransitionAnalyzer,
                                     save: bool = False, filename: Optional[str] = None) -> plt.Figure:
        """
        Visualize P13 transformations at the 7th cycle.
        
        Args:
            analyzer: TransitionAnalyzer instance with P13 transformation data
            save: Whether to save the figure
            filename: Output filename (if save is True)
            
        Returns:
            Matplotlib figure
        """
        # Get P13 seventh cycle transformation data
        p13_data = analyzer.detected_transitions.get('p13_seventh_cycle', [])
        
        if not p13_data:
            logger.warning("No P13 seventh cycle transformation data available")
            fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
            ax.text(0.5, 0.5, "No P13 seventh cycle transformation data available", 
                   ha='center', va='center', fontsize=14)
            plt.close()
            return fig
        
        # Create figure
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        # Phase-Energy plot
        ax1 = fig.add_subplot(221)
        
        # Extract data
        phase_shifts = [t['phase_shift'] for t in p13_data]
        energy_ratios = [t['energy_ratio'] for t in p13_data]
        
        # Color points by phi resonance
        colors = []
        for t in p13_data:
            if t['phi_phase_resonance'] and t['phi_energy_resonance']:
                colors.append('green')  # Both resonances
            elif t['phi_phase_resonance']:
                colors.append('blue')   # Phase resonance only
            elif t['phi_energy_resonance']:
                colors.append('red')    # Energy resonance only
            else:
                colors.append('gray')   # No resonance
        
        # Plot phase shifts vs energy ratios
        ax1.scatter(phase_shifts, energy_ratios, c=colors, alpha=0.7, edgecolor='black')
        
        # Add reference lines for phi and 1/phi
        ax1.axhline(y=self.phi, color='red', linestyle='--', alpha=0.5, label=f'φ = {self.phi:.3f}')
        ax1.axvline(x=1/self.phi, color='blue', linestyle='--', alpha=0.5, label=f'1/φ = {1/self.phi:.3f}')
        ax1.axvline(x=-1/self.phi, color='blue', linestyle=':', alpha=0.5, label=f'-1/φ = {-1/self.phi:.3f}')
        
        ax1.set_xlabel('Phase Shift', fontsize=10)
        ax1.set_ylabel('Energy Ratio', fontsize=10)
        ax1.set_title('P13 Transformation: Phase Shift vs Energy Ratio', fontsize=12)
        ax1.legend(loc='upper right', fontsize=8)
        
        # Time evolution plot
        ax2 = fig.add_subplot(222)
        
        # Sort by tick
        sorted_data = sorted(p13_data, key=lambda x: x['from_tick'])
        
        # Extract time data
        ticks = [t['from_tick'] for t in sorted_data]
        phase_shift_time = [t['phase_shift'] for t in sorted_data]
        energy_ratio_time = [t['energy_ratio'] for t in sorted_data]
        
        # Plot time evolution
        ax2.plot(ticks, phase_shift_time, 'o-', color='blue', label='Phase Shift')
        ax2.set_xlabel('System Tick', fontsize=10)
        ax2.set_ylabel('Phase Shift', fontsize=10, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Add second y-axis for energy ratio
        ax2b = ax2.twinx()
        ax2b.plot(ticks, energy_ratio_time, 'o-', color='red', label='Energy Ratio')
        ax2b.set_ylabel('Energy Ratio', fontsize=10, color='red')
        ax2b.tick_params(axis='y', labelcolor='red')
        
        # Add phi reference lines
        ax2b.axhline(y=self.phi, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=1/self.phi, color='blue', linestyle='--', alpha=0.5)
        ax2.axhline(y=-1/self.phi, color='blue', linestyle=':', alpha=0.5)
        
        # Create combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        ax2.set_title('P13 Transformation: Time Evolution', fontsize=12)
        
        # Recursion depth analysis
        ax3 = fig.add_subplot(212)
        
        # Group data by recursion depth
        data_by_depth = defaultdict(list)
        for t in p13_data:
            depth = t['recursion_depth_before']
            data_by_depth[depth].append(t)
        
        # Calculate metrics for each depth
        depths = sorted(data_by_depth.keys())
        avg_phase_shifts = []
        avg_energy_ratios = []
        phi_resonance_percentages = []
        
        for depth in depths:
            depth_data = data_by_depth[depth]
            avg_phase_shifts.append(np.mean([t['phase_shift'] for t in depth_data]))
            avg_energy_ratios.append(np.mean([t['energy_ratio'] for t in depth_data]))
            
            # Calculate phi resonance percentage
            phi_count = sum(1 for t in depth_data if t['phi_phase_resonance'] or t['phi_energy_resonance'])
            phi_resonance_percentages.append(phi_count / len(depth_data))
        
        # Create grouped bar chart
        bar_width = 0.25
        index = np.arange(len(depths))
        
        ax3.bar(index - bar_width, avg_phase_shifts, bar_width, label='Avg Phase Shift', color='blue', alpha=0.7)
        ax3.bar(index, avg_energy_ratios, bar_width, label='Avg Energy Ratio', color='red', alpha=0.7)
        ax3.bar(index + bar_width, phi_resonance_percentages, bar_width, label='Phi Resonance %', color='green', alpha=0.7)
        
        # Add phi reference line
        ax3.axhline(y=self.phi, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=1/self.phi, color='blue', linestyle='--', alpha=0.5)
        
        ax3.set_xlabel('Recursion Depth', fontsize=10)
        ax3.set_ylabel('Value', fontsize=10)
        ax3.set_xticks(index)
        ax3.set_xticklabels(depths)
        ax3.set_title('P13 Transformation: Metrics by Recursion Depth', fontsize=12)
        ax3.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            if filename is None:
                filename = "p13_seventh_cycle_visualization.png"
            plt.savefig(filename, dpi=self.dpi)
            logger.info(f"Saved visualization to {filename}")
        
        return fig
    
    def visualize_fractal_metrics(self, analyzer: FractalAnalyzer,
                                   save: bool = False, filename: Optional[str] = None) -> plt.Figure:
        """
        Visualize fractal metrics across recursion depths.
        
        Args:
            analyzer: FractalAnalyzer instance with fractal metrics
            save: Whether to save the figure
            filename: Output filename (if save is True)
            
        Returns:
            Matplotlib figure
        """
        # Check if metrics are available
        if not analyzer.fractal_metrics:
            logger.warning("No fractal metrics available")
            fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
            ax.text(0.5, 0.5, "No fractal metrics available", 
                   ha='center', va='center', fontsize=14)
            plt.close()
            return fig
        
        # Create figure
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        # Fractal dimension plot
        ax1 = fig.add_subplot(221)
        
        # Plot fractal dimension calculation if available
        if 'fractal_dimension' in analyzer.fractal_metrics:
            fd_data = analyzer.fractal_metrics['fractal_dimension']
            
            if 'scales' in fd_data and 'counts' in fd_data:
                scales = fd_data['scales']
                counts = fd_data['counts']
                
                # Plot data points
                ax1.scatter(scales, counts, color='blue', alpha=0.7, edgecolor='black')
                
                # Plot best fit line
                if len(scales) > 1:
                    slope, intercept = np.polyfit(scales, counts, 1)
                    x = np.array([min(scales), max(scales)])
                    y = slope * x + intercept
                    ax1.plot(x, y, 'r--', label=f'Slope = {slope:.3f}')
                    
                ax1.set_xlabel('log(1/scale)', fontsize=10)
                ax1.set_ylabel('log(count)', fontsize=10)
                ax1.set_title(f'Fractal Dimension: {fd_data.get("dimension", 0):.3f}', fontsize=12)
                ax1.legend(loc='upper left', fontsize=8)
        else:
            ax1.text(0.5, 0.5, "No fractal dimension data", 
                    ha='center', va='center', fontsize=12)
        
        # Multi-scale entropy plot
        ax2 = fig.add_subplot(222)
        
        # Plot multi-scale entropy if available
        if 'multi_scale_entropy' in analyzer.fractal_metrics:
            mse_data = analyzer.fractal_metrics['multi_scale_entropy']
            
            if 'depths' in mse_data and 'mean_entropy' in mse_data:
                depths = mse_data['depths']
                mean_entropy = mse_data['mean_entropy']
                
                # Plot entropy by depth
                ax2.plot(depths, mean_entropy, 'o-', color='green', alpha=0.7, label='Mean Entropy')
                
                ax2.set_xlabel('Recursion Depth', fontsize=10)
                ax2.set_ylabel('Entropy', fontsize=10)
                ax2.set_title('Multi-Scale Entropy', fontsize=12)
                ax2.legend(loc='upper right', fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No multi-scale entropy data", 
                    ha='center', va='center', fontsize=12)
        
        # Self-similarity plot
        ax3 = fig.add_subplot(223)
        
        # Plot self-similarity if available
        if 'self_similarity' in analyzer.fractal_metrics:
            ss_data = analyzer.fractal_metrics['self_similarity']
            
            if 'similarity_scores' in ss_data:
                similarity_scores = ss_data['similarity_scores']
                
                if similarity_scores:
                    # Create depth labels for x-axis
                    depths = list(range(len(similarity_scores)))
                    depth_labels = [f"{d}-{d+1}" for d in depths]
                    
                    # Plot similarity scores
                    ax3.bar(depths, similarity_scores, color='purple', alpha=0.7, edgecolor='black')
                    
                    ax3.set_xlabel('Depth Transition', fontsize=10)
                    ax3.set_ylabel('Similarity Score', fontsize=10)
                    ax3.set_title(f'Self-Similarity: {ss_data.get("self_similarity", 0):.3f}', fontsize=12)
                    ax3.set_xticks(depths)
                    ax3.set_xticklabels(depth_labels, rotation=45)
        else:
            ax3.text(0.5, 0.5, "No self-similarity data", 
                    ha='center', va='center', fontsize=12)
        
        # Scaling invariance plot
        ax4 = fig.add_subplot(224)
        
        # Plot scaling invariance if available
        if 'scaling_invariance' in analyzer.fractal_metrics:
            si_data = analyzer.fractal_metrics['scaling_invariance']
            
            if 'invariant_positions' in si_data:
                # Create position distribution
                positions = list(range(1, 14))  # Positions 1-13
                
                # Mark invariant positions
                invariant = [1 if p in si_data['invariant_positions'] else 0 for p in positions]
                
                # Create bar chart
                ax4.bar(positions, invariant, color='orange', alpha=0.7, edgecolor='black')
                
                ax4.set_xlabel('Position Number', fontsize=10)
                ax4.set_ylabel('Invariant', fontsize=10)
                ax4.set_title('Scaling-Invariant Positions', fontsize=12)
                ax4.set_yticks([0, 1])
                ax4.set_yticklabels(['No', 'Yes'])
                ax4.set_xticks(positions)
        else:
            ax4.text(0.5, 0.5, "No scaling invariance data", 
                    ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            if filename is None:
                filename = "fractal_metrics_visualization.png"
            plt.savefig(filename, dpi=self.dpi)
            logger.info(f"Saved visualization to {filename}")
        
        return fig 