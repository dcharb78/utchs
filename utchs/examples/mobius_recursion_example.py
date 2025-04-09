"""
Example demonstrating OmniLens Möbius recursion integration with UTCHS framework.

This example shows how the recursive phase dynamics from the OmniLens Möbius Framework
can be used with the UTCHS phase field system for enhanced phase recursion modeling.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Add parent directory to path to import UTCHS modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utchs.math.mobius import MobiusNode, MobiusTransformation
from utchs.fields.resonant_field import ResonantField
from utchs.math.phase_field import PhaseField
from utchs.core.position import Position


def run_basic_mobius_node_example():
    """Demonstrate basic MobiusNode recursion with a ResonantField."""
    # Create a resonant field with default tuning
    field = ResonantField(tuning=144000)
    
    # Create a Möbius node with the golden ratio as signal
    node = MobiusNode(signal=complex(1.618, 0), phase=0)
    
    # Create a simple Möbius transformation
    mobius = MobiusTransformation(
        a=complex(0.8, 0.1),
        b=complex(0.2, 0.3),
        c=complex(-0.1, 0.1),
        d=complex(1.0, 0.0)
    )
    
    # Run the recursion for several iterations
    print("MobiusNode Recursion Example:")
    print("-----------------------------")
    print(f"Initial Signal: {node.signal}, Phase: {node.phase}")
    
    coherence_values = []
    phase_values = []
    signal_values = []
    
    for i in range(10):
        # Apply the Möbius transformation
        node.apply_mobius(mobius.a, mobius.b, mobius.c, mobius.d)
        
        # Calculate field resonance at the center of the field
        coherence = field.apply_to_node(node, (0, 0, 0))
        
        # Store values for plotting
        coherence_values.append(coherence)
        phase_values.append(node.phase)
        signal_values.append(abs(node.signal))
        
        # Print results
        print(f"Iteration {i+1}: Signal: {node.signal:.4f}, Phase: {node.phase:.4f}, Coherence: {coherence:.4f}")
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(coherence_values, 'o-', color='blue')
    plt.title('Coherence Values')
    plt.xlabel('Iteration')
    plt.ylabel('Coherence')
    
    plt.subplot(1, 3, 2)
    plt.plot(phase_values, 'o-', color='red')
    plt.title('Phase Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Phase (radians)')
    
    plt.subplot(1, 3, 3)
    plt.plot(signal_values, 'o-', color='green')
    plt.title('Signal Magnitude')
    plt.xlabel('Iteration')
    plt.ylabel('|Signal|')
    
    plt.tight_layout()
    plt.savefig('mobius_node_recursion.png')


def run_phase_field_integration_example():
    """Demonstrate integration between MobiusNode network and PhaseField."""
    # Create configuration for phase field
    config = {
        'grid_size': (32, 32, 1),
        'grid_spacing': 0.1,
        'initial_pattern': 'vortex',
        'history_length': 10
    }
    
    # Create phase field
    phase_field = PhaseField(config)
    
    # Create resonant field
    resonant_field = ResonantField(tuning=144000, dimensions=3, resolution=32)
    
    # Create a network of MobiusNodes corresponding to phase field cells
    nodes = {}
    grid_shape = config['grid_size']
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            # Extract phase and amplitude from phase field
            cell_value = phase_field.field[i, j, 0]
            amplitude = abs(cell_value)
            phase = np.angle(cell_value)
            
            # Create node with field value as signal
            nodes[(i, j)] = MobiusNode(signal=cell_value, phase=phase)
    
    # Create a Möbius transformation
    mobius = MobiusTransformation(
        a=complex(0.9, 0.1),
        b=complex(0.0, 0.2),
        c=complex(-0.05, 0.05),
        d=complex(1.0, 0.0)
    )
    
    # Set up plot for visualization
    plt.figure(figsize=(12, 10))
    
    # Initial phase field
    plt.subplot(2, 2, 1)
    plt.imshow(np.angle(phase_field.field[:, :, 0]), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(label='Phase')
    plt.title('Initial Phase Field')
    
    # Apply recursion to nodes and update phase field
    coherence_field = np.zeros(grid_shape[:2])
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            node = nodes[(i, j)]
            
            # Apply Möbius transformation
            node.apply_mobius(mobius.a, mobius.b, mobius.c, mobius.d)
            
            # Calculate normalized position in field space
            pos_x = 2 * i / grid_shape[0] - 1
            pos_y = 2 * j / grid_shape[1] - 1
            pos_z = 0
            
            # Apply field effects
            coherence = resonant_field.apply_to_node(node, (pos_x, pos_y, pos_z))
            coherence_field[i, j] = coherence
            
            # Update phase field with new signal
            phase_field.field[i, j, 0] = node.signal
    
    # Updated phase field
    plt.subplot(2, 2, 2)
    plt.imshow(np.angle(phase_field.field[:, :, 0]), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(label='Phase')
    plt.title('Updated Phase Field')
    
    # Coherence field
    plt.subplot(2, 2, 3)
    plt.imshow(coherence_field, cmap='viridis')
    plt.colorbar(label='Coherence')
    plt.title('Coherence Field')
    
    # Signal amplitude
    amplitude_field = np.abs(phase_field.field[:, :, 0])
    plt.subplot(2, 2, 4)
    plt.imshow(amplitude_field, cmap='plasma')
    plt.colorbar(label='Amplitude')
    plt.title('Signal Amplitude')
    
    plt.tight_layout()
    plt.savefig('phase_field_integration.png')


def run_utchs_position_mapping_example():
    """Demonstrate mapping between UTCHS positions and the OmniLens framework."""
    # Create positions from the UTCHS framework for all 13 positions
    positions = [Position(i) for i in range(1, 14)]
    
    # Create a resonant field
    field = ResonantField(tuning=144000)
    
    # Map each UTCHS position to a MobiusNode
    position_nodes = {}
    
    for pos in positions:
        # Create a MobiusNode with position-specific properties
        # Map position number to a phase value in [0, 2π]
        phase = (pos.number - 1) / 13 * 2 * np.pi
        
        # Special handling for specific positions based on UTCHS theory
        if pos.number in [3, 6, 9]:  # Vortex positions
            # Use a specific "vortex" signal format with enhanced magnitude
            signal = complex(1.2 * np.cos(phase), 1.2 * np.sin(phase))
        elif pos.number in [1, 2, 3, 5, 7, 11, 13]:  # Prime positions
            # Use a "prime" signal format with phase-derived real component
            signal = complex(1.0, 0.5 * np.sin(2 * phase))
        else:
            # Regular signal
            signal = complex(np.cos(phase), np.sin(phase))
            
        # Create node
        node = MobiusNode(signal=signal, phase=phase)
        position_nodes[pos.number] = node
    
    # Apply resonance calculation based on position properties
    results = []
    for pos in positions:
        node = position_nodes[pos.number]
        
        # Calculate a position-specific field index
        # Map the 13 positions to coordinates in the field
        theta = (pos.number - 1) / 13 * 2 * np.pi
        x = np.cos(theta)
        y = np.sin(theta)
        z = 0  # Use 2D mapping for simplicity
        
        # Apply field resonance
        coherence = field.apply_to_node(node, (x, y, z))
        
        results.append({
            'position': pos.number,
            'role': pos.role,
            'signal': node.signal,
            'phase': node.phase,
            'coherence': coherence
        })
    
    # Print and plot the results
    print("\nUTCHS Position Mapping Example:")
    print("-------------------------------")
    for result in results:
        print(f"Position {result['position']} ({result['role']}): "
              f"Coherence = {result['coherence']:.4f}, "
              f"Phase = {result['phase']:.4f}")
    
    # Plot position coherence values
    plt.figure(figsize=(12, 8))
    
    # Plot on a circle to show the cyclical nature
    angles = [(pos.number - 1) / 13 * 2 * np.pi for pos in positions]
    coherence_values = [result['coherence'] for result in results]
    
    # Polar plot
    plt.subplot(1, 2, 1, projection='polar')
    plt.plot(angles, coherence_values, 'o-', linewidth=2)
    
    # Add position labels
    for i, pos in enumerate(positions):
        angle = angles[i]
        radius = coherence_values[i]
        plt.annotate(str(pos.number), 
                     xy=(angle, radius),
                     xytext=(angle, radius + 0.05),
                     ha='center')
    
    plt.title('UTCHS Position Coherence (Polar)')
    
    # Bar chart highlighting position types
    plt.subplot(1, 2, 2)
    bars = plt.bar(range(1, 14), coherence_values)
    
    # Color-code bars by position type
    for i, pos in enumerate(positions):
        if pos.number in [3, 6, 9]:  # Vortex positions
            bars[i].set_color('red')
        elif pos.number in [1, 2, 3, 5, 7, 11, 13]:  # Prime positions
            # Avoid duplicating position 3 since it's also a vortex position
            if pos.number not in [3]:
                bars[i].set_color('green')
        else:
            bars[i].set_color('blue')
    
    plt.xlabel('Position Number')
    plt.ylabel('Coherence')
    plt.title('UTCHS Position Coherence by Type')
    plt.xticks(range(1, 14))
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Vortex Positions (3,6,9)'),
        Patch(facecolor='green', label='Prime Positions (1,2,5,7,11,13)'),
        Patch(facecolor='blue', label='Regular Positions')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('utchs_position_mapping.png')


def run_phase_recursion_animation():
    """Create an animation of phase recursion using the OmniLens concepts."""
    # Create configuration for phase field
    config = {
        'grid_size': (64, 64, 1),
        'grid_spacing': 0.1,
        'initial_pattern': 'spiral',
        'history_length': 20
    }
    
    # Create phase field
    phase_field = PhaseField(config)
    
    # Create resonant field
    resonant_field = ResonantField(tuning=144000, dimensions=3, resolution=64)
    
    # Create a Möbius transformation
    mobius = MobiusTransformation(
        a=complex(0.99, 0.01),
        b=complex(0.0, 0.1),
        c=complex(-0.01, 0.01),
        d=complex(1.0, 0.0)
    )
    
    # Create figure for animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Initialize plots
    phase_plot = ax1.imshow(np.angle(phase_field.field[:, :, 0]), 
                           cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax1.set_title('Phase Field')
    plt.colorbar(phase_plot, ax=ax1, label='Phase')
    
    coherence_plot = ax2.imshow(np.zeros(config['grid_size'][:2]), 
                               cmap='viridis', vmin=0, vmax=1)
    ax2.set_title('Coherence Field')
    plt.colorbar(coherence_plot, ax=ax2, label='Coherence')
    
    plt.tight_layout()
    
    # Create a network of MobiusNodes corresponding to phase field cells
    nodes = {}
    grid_shape = config['grid_size']
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            # Extract phase and amplitude from phase field
            cell_value = phase_field.field[i, j, 0]
            nodes[(i, j)] = MobiusNode(signal=cell_value, phase=np.angle(cell_value))
    
    # Animation update function
    def update(frame):
        # Apply recursion to nodes and update phase field
        coherence_field = np.zeros(grid_shape[:2])
        
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                node = nodes[(i, j)]
                
                # Apply Möbius transformation
                node.apply_mobius(mobius.a, mobius.b, mobius.c, mobius.d)
                
                # Calculate normalized position in field space
                pos_x = 2 * i / grid_shape[0] - 1
                pos_y = 2 * j / grid_shape[1] - 1
                pos_z = 0
                
                # Apply field effects
                coherence = resonant_field.apply_to_node(node, (pos_x, pos_y, pos_z))
                coherence_field[i, j] = coherence
                
                # Update phase field with new signal
                phase_field.field[i, j, 0] = node.signal
        
        # Update plots
        phase_plot.set_array(np.angle(phase_field.field[:, :, 0]))
        coherence_plot.set_array(coherence_field)
        
        return phase_plot, coherence_plot
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=range(50), blit=True, interval=100)
    
    # Save animation
    ani.save('phase_recursion.gif', writer='pillow', fps=10)
    
    print("Animation saved as 'phase_recursion.gif'")


if __name__ == "__main__":
    print("Running OmniLens Möbius Framework Integration Examples")
    print("=====================================================\n")
    
    # Run the examples
    run_basic_mobius_node_example()
    run_phase_field_integration_example()
    run_utchs_position_mapping_example()
    run_phase_recursion_animation()
    
    print("\nAll examples completed successfully!")
    print("Check the generated images and animations for visualization of the results.") 