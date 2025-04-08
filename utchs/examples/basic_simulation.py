"""
Basic simulation example for the UTCHS framework.

This script demonstrates how to set up and run a simulation of the Enhanced 
Unified Toroidal-Crystalline Harmonic System (UTCHS), visualize the results,
and analyze the system dynamics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yaml
import argparse
from datetime import datetime

# Add parent directory to path to import UTCHS modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utchs.config import load_config
from utchs.core.system import UTCHSSystem
from utchs.visualization.field_vis import FieldVisualizer
from utchs.visualization.torus_vis import TorusVisualizer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run UTCHS simulation')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ticks', type=int, default=100, help='Number of ticks to simulate')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--checkpoint', type=str, default=None, help='Load from checkpoint file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--analysis', action='store_true', help='Perform detailed analysis')
    return parser.parse_args()


def setup_output_directory(dir_name):
    """Create output directory with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(dir_name, f'utchs_sim_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_simulation(config, ticks, checkpoint=None):
    """
    Run the UTCHS simulation.
    
    Args:
        config: Configuration dictionary
        ticks: Number of ticks to simulate
        checkpoint: Optional path to checkpoint file
        
    Returns:
        UTCHSSystem instance after simulation
    """
    print(f"Initializing UTCHS system...")
    
    # Create system
    system = UTCHSSystem(config)
    
    # Load from checkpoint if specified
    if checkpoint:
        print(f"Loading from checkpoint: {checkpoint}")
        system.load_checkpoint(checkpoint)
    
    # Run simulation
    print(f"Running simulation for {ticks} ticks...")
    states = system.run_simulation(ticks)
    print(f"Simulation completed.")
    
    return system


def visualize_results(system, output_dir):
    """
    Generate visualizations of the simulation results.
    
    Args:
        system: UTCHSSystem instance
        output_dir: Output directory path
    """
    print(f"Generating visualizations in {output_dir}...")
    
    # Create visualization directories
    viz_dir = os.path.join(output_dir, 'visualizations')
    field_dir = os.path.join(viz_dir, 'fields')
    torus_dir = os.path.join(viz_dir, 'torus')
    analysis_dir = os.path.join(viz_dir, 'analysis')
    
    os.makedirs(field_dir, exist_ok=True)
    os.makedirs(torus_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Create visualizers
    field_vis = FieldVisualizer(output_dir=field_dir)
    torus_vis = TorusVisualizer(output_dir=torus_dir)
    
    # Visualize phase field
    print("Visualizing phase field...")
    for axis in range(3):
        field_vis.visualize_phase_slice(
            system.phase_field,
            axis=axis,
            save=True,
            filename=f'phase_slice_axis{axis}_tick{system.current_tick}.png'
        )
    
    # Visualize energy field
    print("Visualizing energy field...")
    for axis in range(3):
        field_vis.visualize_energy_slice(
            system.energy_field,
            axis=axis,
            save=True,
            filename=f'energy_slice_axis{axis}_tick{system.current_tick}.png'
        )
    
    # Visualize torus structures
    print("Visualizing toroidal structures...")
    for torus in system.tori:
        torus_vis.visualize_torus(
            torus,
            save=True,
            filename=f'torus_{torus.id}_tick{system.current_tick}.png'
        )
    
    # Visualize position network
    print("Visualizing position network...")
    torus_vis.visualize_position_network(
        system,
        save=True,
        filename=f'position_network_tick{system.current_tick}.png'
    )
    
    # Create simple animation of field evolution
    if len(system.phase_field.field_history) > 1:
        print("Creating field evolution animation...")
        field_vis.create_animation(
            system.phase_field.field_history,
            slice_axis=2,
            fps=10,
            filename=f'phase_evolution_tick{system.current_tick}.mp4'
        )
    
    print("Visualization complete.")


def analyze_system(system, output_dir):
    """
    Perform detailed analysis of the system state.
    
    Args:
        system: UTCHSSystem instance
        output_dir: Output directory path
    """
    print("Performing system analysis...")
    
    # Create analysis directory
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Get system analysis
    analysis = system.analyze_system_state()
    
    # Save analysis to file
    analysis_file = os.path.join(analysis_dir, f'analysis_tick{system.current_tick}.yaml')
    with open(analysis_file, 'w') as f:
        yaml.dump(analysis, f, default_flow_style=False)
    
    # Generate analysis plots
    print("Generating analysis plots...")
    
    # Plot hierarchy metrics
    plt.figure(figsize=(10, 6))
    torus_coherence = [t["phase_coherence"] for t in analysis["hierarchy_metrics"]["torus_metrics"]]
    torus_stability = [t["stability"] for t in analysis["hierarchy_metrics"]["torus_metrics"]]
    torus_ids = [t["id"] for t in analysis["hierarchy_metrics"]["torus_metrics"]]
    
    plt.bar(torus_ids, torus_coherence, alpha=0.7, label='Phase Coherence')
    plt.bar(torus_ids, torus_stability, alpha=0.7, label='Stability')
    plt.xlabel('Torus ID')
    plt.ylabel('Metric Value')
    plt.title('Torus Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(analysis_dir, f'torus_metrics_tick{system.current_tick}.png'), dpi=300)
    
    # Plot position distribution
    plt.figure(figsize=(12, 6))
    pos_percentages = analysis["hierarchy_metrics"]["position_distribution"]["percentages"]
    positions = list(pos_percentages.keys())
    percentages = list(pos_percentages.values())
    
    plt.bar(positions, percentages, color='skyblue')
    
    # Highlight special positions
    vortex_positions = [3, 6, 9]
    prime_positions = [1, 2, 3, 5, 7, 11, 13]
    
    for i, pos in enumerate(positions):
        if pos in vortex_positions:
            plt.bar(pos, percentages[i], color='red', alpha=0.7)
        elif pos in prime_positions and pos not in vortex_positions:
            plt.bar(pos, percentages[i], color='green', alpha=0.7)
    
    plt.xlabel('Position Number')
    plt.ylabel('Percentage (%)')
    plt.title('Position Distribution')
    plt.xticks(positions)
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='Regular Positions'),
        Patch(facecolor='green', alpha=0.7, label='Prime Positions'),
        Patch(facecolor='red', alpha=0.7, label='Vortex Positions')
    ]
    plt.legend(handles=legend_elements)
    
    plt.savefig(os.path.join(analysis_dir, f'position_distribution_tick{system.current_tick}.png'), dpi=300)
    
    # Plot phase recursion metrics
    plt.figure(figsize=(10, 6))
    x = ['Recursion Depth', 'Field Complexity', 'Self Similarity']
    y = [
        analysis["phase_recursion"]["recursion_depth"],
        analysis["phase_recursion"]["field_complexity"],
        analysis["phase_recursion"]["self_similarity"]
    ]
    
    plt.bar(x, y, color=['purple', 'orange', 'teal'])
    plt.ylabel('Metric Value')
    plt.title('Phase Recursion Metrics')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(analysis_dir, f'phase_recursion_metrics_tick{system.current_tick}.png'), dpi=300)
    
    print(f"Analysis complete. Results saved to {analysis_dir}")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Load configuration
    config = load_config(args.config)
    
    # Save configuration copy
    config_file = os.path.join(output_dir, 'config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run simulation
    system = run_simulation(config, args.ticks, args.checkpoint)
    
    # Generate visualizations if requested
    if args.visualize:
        visualize_results(system, output_dir)
    
    # Perform detailed analysis if requested
    if args.analysis:
        analyze_system(system, output_dir)
    
    # Save final checkpoint
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f'utchs_final_tick{system.current_tick}.npz')
    
    try:
        system._save_checkpoint()
        print(f"Final checkpoint saved to {checkpoint_file}")
    except Exception as e:
        print(f"Failed to save checkpoint: {str(e)}")
    
    print(f"Simulation results saved to {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()