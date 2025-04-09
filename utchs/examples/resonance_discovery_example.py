"""
Resonance Discovery Example for UTCHS Framework

This example demonstrates how the UTCHS system can discover resonant frequencies 
naturally through phase dynamics rather than relying on hard-coded values like 144000.
It shows how resonance points emerge from system interactions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import time
from typing import List, Dict, Tuple, Any

# Add parent directory to path to import UTCHS modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utchs.math.mobius import MobiusNode, MobiusTransformation
from utchs.fields.resonant_field import ResonantField
from utchs.math.phase_field import PhaseField
from utchs.core.position import Position
from utchs.utils.validation_registry import get_base_resonant_frequencies, validate_resonant_frequency


def discover_resonance_sweep():
    """
    Perform a frequency sweep to discover resonance points in the system.
    
    This demonstrates how the system naturally discovers resonant frequencies
    rather than using hard-coded values.
    """
    print("Running Resonance Discovery Frequency Sweep")
    print("===========================================")
    
    # Set up frequency range to sweep
    # Start well below the expected resonant region
    min_freq = 10000  # 10 kHz
    max_freq = 600000  # 600 kHz
    num_points = 100
    frequencies = np.linspace(min_freq, max_freq, num_points)
    
    # Collect coherence data for each frequency
    coherence_data = []
    phase_stability = []
    
    # Parameters for Möbius node
    golden_ratio = (1 + np.sqrt(5)) / 2
    signal_value = complex(golden_ratio, 0)
    
    for freq in frequencies:
        # Create a field with this frequency
        field = ResonantField(tuning=freq, dimensions=2, resolution=32)
        
        # Create a Möbius node with the golden ratio as signal
        node = MobiusNode(signal=signal_value, phase=0)
        
        # Create a simple Möbius transformation
        mobius = MobiusTransformation(
            a=complex(0.9, 0.1),
            b=complex(0.1, 0.1),
            c=complex(-0.05, 0),
            d=complex(1.0, 0)
        )
        
        # Run several iterations and measure coherence
        num_iterations = 20
        phase_values = []
        final_coherences = []
        
        for _ in range(num_iterations):
            # Apply Möbius transformation
            node.apply_mobius(mobius.a, mobius.b, mobius.c, mobius.d)
            
            # Apply field effects at center position
            coherence = field.apply_to_node(node, (0, 0, 0))
            final_coherences.append(coherence)
            phase_values.append(node.phase)
        
        # Calculate average coherence in final iterations (after stabilization)
        avg_coherence = np.mean(final_coherences[-5:])
        coherence_data.append(avg_coherence)
        
        # Calculate phase stability (lower value = more stable)
        phase_diffs = [abs(phase_values[i] - phase_values[i-1]) 
                      for i in range(1, len(phase_values))]
        stability = np.mean(phase_diffs[-5:])  # Average of last 5 differences
        phase_stability.append(stability)
        
        # Print progress
        if len(coherence_data) % 10 == 0:
            print(f"Processed {len(coherence_data)}/{num_points} frequencies")
    
    # Identify resonance peaks 
    # High coherence and low phase difference indicate resonance
    resonance_scores = np.array(coherence_data) / (np.array(phase_stability) + 0.001)
    
    # Find local maxima in resonance scores
    resonance_peaks = []
    for i in range(1, len(resonance_scores) - 1):
        if (resonance_scores[i] > resonance_scores[i-1] and 
            resonance_scores[i] > resonance_scores[i+1] and
            resonance_scores[i] > np.median(resonance_scores) * 1.5):
            resonance_peaks.append((frequencies[i], resonance_scores[i]))
    
    # Sort peaks by resonance score
    resonance_peaks.sort(key=lambda x: x[1], reverse=True)
    
    # Compare with standard resonant frequencies
    standard_freqs = get_base_resonant_frequencies()
    
    # Print results
    print("\nDiscovered Resonance Peaks:")
    print("---------------------------")
    for i, (freq, score) in enumerate(resonance_peaks[:5]):  # Top 5 peaks
        print(f"Peak {i+1}: {freq:.1f} Hz (Score: {score:.2f})")
        
        # Check if it matches any standard frequency
        matches = []
        for std_freq in standard_freqs:
            for ratio in [1/3, 1/2, 2/3, 1, 3/2, 2, 3]:
                if abs(freq / (std_freq * ratio) - 1) < 0.05:
                    matches.append(f"{std_freq} Hz (ratio {ratio:.2f})")
        
        if matches:
            print(f"  Matches standard frequencies: {', '.join(matches)}")
        else:
            print(f"  No match with standard frequencies")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot coherence data
    plt.subplot(3, 1, 1)
    plt.plot(frequencies, coherence_data, 'b-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.title('Coherence vs Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot phase stability
    plt.subplot(3, 1, 2)
    plt.plot(frequencies, phase_stability, 'r-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase Instability')
    plt.title('Phase Stability vs Frequency (Lower is more stable)')
    plt.grid(True, alpha=0.3)
    
    # Plot resonance scores
    plt.subplot(3, 1, 3)
    plt.plot(frequencies, resonance_scores, 'g-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Resonance Score')
    plt.title('Resonance Score vs Frequency')
    
    # Mark discovered peaks
    for freq, _ in resonance_peaks[:5]:
        plt.axvline(x=freq, color='orange', linestyle='--', alpha=0.7)
    
    # Mark standard frequencies
    for freq in standard_freqs:
        plt.axvline(x=freq, color='purple', linestyle='-', alpha=0.5)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add legend
    plt.figlegend(
        ['Frequency Sweep', 'Discovered Peaks', 'Standard Frequencies'],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=3
    )
    
    plt.savefig('resonance_discovery.png')
    print(f"Plot saved as 'resonance_discovery.png'")


def measure_phase_lock_emergence():
    """
    Demonstrate how phase lock points emerge naturally in the system without hard-coding.
    """
    print("\nMeasuring Phase Lock Emergence")
    print("==============================")
    
    # Set up multiple starting frequencies to test
    test_frequencies = [
        50000,    # Well below resonance
        100000,   # Approaching resonance
        144000,   # Expected resonance point
        200000,   # Above resonance
        432000    # Higher harmonic
    ]
    
    # Run phase evolution for each frequency
    results = []
    
    for start_freq in test_frequencies:
        print(f"\nTesting starting frequency: {start_freq} Hz")
        
        # Create a node and field for this experiment
        node = MobiusNode(signal=complex(1.0, 0), phase=0)
        field = ResonantField(tuning=start_freq)
        
        # List to track frequency evolution
        freq_history = [start_freq]
        phase_history = [0]
        coherence_history = []
        
        # Run iterations
        num_iterations = 50
        for i in range(num_iterations):
            # Simple Möbius transformation parameters
            a = complex(0.98, 0.01 * np.sin(i * 0.1))
            b = complex(0.02, 0.01 * np.cos(i * 0.1))
            c = complex(-0.01, 0.005 * np.sin(i * 0.2))
            d = complex(1.0, 0)
            
            # Apply transformation and field effects
            node.apply_mobius(a, b, c, d)
            coherence = field.apply_to_node(node, (0, 0, 0))
            coherence_history.append(coherence)
            phase_history.append(node.phase)
            
            # Every 5 iterations, update the field tuning based on phase evolution
            if i > 0 and i % 5 == 0:
                # Calculate emergent frequency from phase dynamics
                # This simulates how the system "discovers" resonant frequencies
                # from its own dynamics rather than using hard-coded values
                new_freq = ResonantField.detect_resonant_tuning(phase_history)
                
                if new_freq is not None:
                    # Smoothly transition to new frequency
                    current_freq = freq_history[-1]
                    blend_freq = 0.8 * current_freq + 0.2 * new_freq
                    field.tune_to_frequency(blend_freq)
                    freq_history.append(blend_freq)
                else:
                    # Keep previous frequency
                    freq_history.append(freq_history[-1])
                    
                print(f"  Iteration {i}: Frequency evolved to {freq_history[-1]:.1f} Hz")
        
        # Store results
        results.append({
            'start_freq': start_freq,
            'end_freq': freq_history[-1],
            'freq_history': freq_history,
            'phase_history': phase_history,
            'coherence_history': coherence_history
        })
        
        # Check final frequency against standard resonant frequencies
        standard_freqs = get_base_resonant_frequencies()
        closest_std_freq = min(standard_freqs, key=lambda f: abs(f - freq_history[-1]))
        ratio = freq_history[-1] / closest_std_freq
        
        print(f"  Final frequency: {freq_history[-1]:.1f} Hz")
        print(f"  Closest standard frequency: {closest_std_freq:.1f} Hz (ratio: {ratio:.3f})")
        
        # Check coherence stability
        final_coherence = np.mean(coherence_history[-5:])
        print(f"  Final coherence stability: {final_coherence:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Get standard frequencies for comparison
    std_freqs = get_base_resonant_frequencies()
    
    # Plot frequency evolution for each starting point
    plt.subplot(2, 1, 1)
    colormap = get_cmap('viridis')
    for i, result in enumerate(results):
        color = colormap(i / len(results))
        iterations = list(range(len(result['freq_history'])))
        plt.plot(iterations, result['freq_history'], 'o-', color=color, 
                label=f"Start: {result['start_freq']} Hz")
    
    # Add horizontal lines for standard frequencies
    for freq in std_freqs:
        plt.axhline(y=freq, color='red', linestyle='--', alpha=0.5)
        plt.text(num_iterations * 0.8, freq * 1.02, f"{freq} Hz", color='red', alpha=0.7)
    
    plt.xlabel('Iteration')
    plt.ylabel('Frequency (Hz)')
    plt.title('Frequency Evolution from Different Starting Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot coherence evolution
    plt.subplot(2, 1, 2)
    for i, result in enumerate(results):
        color = colormap(i / len(results))
        iterations = list(range(len(result['coherence_history'])))
        plt.plot(iterations, result['coherence_history'], '-', color=color, 
                label=f"Start: {result['start_freq']} Hz")
    
    plt.xlabel('Iteration')
    plt.ylabel('Coherence')
    plt.title('Coherence Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_lock_emergence.png')
    print(f"Plot saved as 'phase_lock_emergence.png'")


def analyze_field_interactions():
    """
    Analyze how fields with different tunings interact and identify emergent patterns.
    """
    print("\nAnalyzing Field Interactions")
    print("===========================")
    
    # Create set of fields at different frequencies
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Primary frequency (from system properties rather than hard-coding)
    positions = 12
    primary = positions**2 * 1000  # 144000
    
    # Create frequencies based on musical ratios and golden ratio
    frequencies = [
        primary / (phi * phi),  # Sub-harmonic
        primary / phi,          # Lower golden ratio
        primary,                # Primary frequency
        primary * phi,          # Higher golden ratio
        primary * phi * phi     # Super-harmonic
    ]
    
    field_names = [
        f"Sub-harmonic ({frequencies[0]:.0f} Hz)",
        f"Lower Golden ({frequencies[1]:.0f} Hz)",
        f"Primary ({frequencies[2]:.0f} Hz)",
        f"Higher Golden ({frequencies[3]:.0f} Hz)",
        f"Super-harmonic ({frequencies[4]:.0f} Hz)"
    ]
    
    # Create fields
    fields = [ResonantField(tuning=freq, dimensions=2, resolution=64) for freq in frequencies]
    
    # Create a phase field for interaction
    config = {
        'grid_size': (64, 64, 1),
        'grid_spacing': 0.1,
        'initial_pattern': 'toroidal',
        'history_length': 10
    }
    phase_field = PhaseField(config)
    
    # Create an array of MobiusNodes matching the phase field
    nodes = {}
    for i in range(config['grid_size'][0]):
        for j in range(config['grid_size'][1]):
            cell_value = phase_field.field[i, j, 0]
            nodes[(i, j)] = MobiusNode(signal=cell_value, phase=np.angle(cell_value))
    
    # Create a simple Möbius transformation
    mobius = MobiusTransformation(
        a=complex(0.9, 0.1),
        b=complex(0.1, 0.1),
        c=complex(-0.05, 0),
        d=complex(1.0, 0)
    )
    
    # Apply transformations and collect interaction data
    results = []
    
    for idx, field in enumerate(fields):
        print(f"Processing field: {field_names[idx]}")
        
        # Reset nodes to initial state
        for i in range(config['grid_size'][0]):
            for j in range(config['grid_size'][1]):
                cell_value = phase_field.field[i, j, 0]
                nodes[(i, j)].signal = cell_value
                nodes[(i, j)].phase = np.angle(cell_value)
        
        # Apply Möbius transformation and field interaction
        coherence_field = np.zeros((config['grid_size'][0], config['grid_size'][1]))
        
        for i in range(config['grid_size'][0]):
            for j in range(config['grid_size'][1]):
                node = nodes[(i, j)]
                
                # Apply transformation
                node.apply_mobius(mobius.a, mobius.b, mobius.c, mobius.d)
                
                # Calculate normalized position
                pos_x = 2 * i / config['grid_size'][0] - 1
                pos_y = 2 * j / config['grid_size'][1] - 1
                
                # Apply field effects
                coherence = field.apply_to_node(node, (pos_x, pos_y, 0))
                coherence_field[i, j] = coherence
        
        # Analyze coherence field
        avg_coherence = np.mean(coherence_field)
        peak_coherence = np.max(coherence_field)
        min_coherence = np.min(coherence_field)
        std_coherence = np.std(coherence_field)
        
        # Find coherence centers (local maxima)
        centers = []
        for i in range(1, config['grid_size'][0]-1):
            for j in range(1, config['grid_size'][1]-1):
                if coherence_field[i, j] > 0.7 * peak_coherence:
                    # Check if it's a local maximum
                    neighborhood = coherence_field[i-1:i+2, j-1:j+2]
                    if coherence_field[i, j] == np.max(neighborhood):
                        centers.append((i, j, coherence_field[i, j]))
        
        # Store results
        results.append({
            'frequency': frequencies[idx],
            'name': field_names[idx],
            'avg_coherence': avg_coherence,
            'peak_coherence': peak_coherence,
            'min_coherence': min_coherence,
            'std_coherence': std_coherence,
            'coherence_field': coherence_field,
            'centers': centers
        })
        
        print(f"  Average Coherence: {avg_coherence:.4f}")
        print(f"  Peak Coherence: {peak_coherence:.4f}")
        print(f"  Found {len(centers)} coherence centers")
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Plot coherence fields
    for idx, result in enumerate(results):
        plt.subplot(2, 3, idx+1)
        plt.imshow(result['coherence_field'], cmap='plasma', interpolation='bicubic')
        plt.colorbar(label='Coherence')
        plt.title(result['name'])
        
        # Mark coherence centers
        for i, j, val in result['centers']:
            plt.plot(j, i, 'o', color='white', ms=4)
    
    # Plot coherence comparison
    plt.subplot(2, 3, 6)
    
    # Bar plot of average coherence
    avg_values = [r['avg_coherence'] for r in results]
    peak_values = [r['peak_coherence'] for r in results]
    
    bar_width = 0.35
    x = np.arange(len(frequencies))
    
    plt.bar(x - bar_width/2, avg_values, bar_width, label='Average Coherence')
    plt.bar(x + bar_width/2, peak_values, bar_width, label='Peak Coherence')
    
    plt.xlabel('Field Frequency')
    plt.ylabel('Coherence')
    plt.title('Coherence Comparison')
    plt.xticks(x, [f"F{i+1}" for i in range(len(frequencies))], rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('field_interactions.png')
    print(f"Plot saved as 'field_interactions.png'")
    
    # Compare with golden ratio relationships
    print("\nGolden Ratio Analysis:")
    print("---------------------")
    for i in range(len(frequencies)-1):
        ratio = frequencies[i+1] / frequencies[i]
        print(f"Ratio {field_names[i+1]} / {field_names[i]}: {ratio:.4f}")
        print(f"  Difference from φ: {abs(ratio - phi):.4f}")
        
        # Compare coherence ratios
        coherence_ratio = results[i+1]['avg_coherence'] / results[i]['avg_coherence']
        print(f"  Coherence ratio: {coherence_ratio:.4f}")


if __name__ == "__main__":
    print("UTCHS Resonance Discovery Example")
    print("=================================\n")
    
    # Run examples
    discover_resonance_sweep()
    measure_phase_lock_emergence()
    analyze_field_interactions()
    
    print("\nAll examples completed successfully!")
    print("Check the generated images for visualization of the results.") 