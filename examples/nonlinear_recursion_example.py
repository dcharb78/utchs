#!/usr/bin/env python3
"""
Example script demonstrating the nonlinear recursion enhancements in UTCHS.

This script creates a simulated system and analyzes meta-patterns using the
nonlinear recursion, phase-locking, and coherence gating enhancements.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import json

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utchs.core.system import UTCHSSystem
from utchs.core.recursion_tracker import RecursionTracker
from utchs.core.meta_pattern_detector import MetaPatternDetector
from utchs.mathematics.recursion_scaling import RecursionScaling
from utchs.core.phase_lock import TorsionalPhaseLock
from utchs.core.coherence_gate import CoherenceGate
from utchs.mathematics.mobius import MobiusTransformation
from utchs.core.fractal_analyzer import FractalAnalyzer

def simulate_system(num_ticks: int = 1000) -> UTCHSSystem:
    """
    Run a simulation of the UTCHS system.
    
    Args:
        num_ticks: Number of ticks to simulate
        
    Returns:
        Simulated UTCHSSystem
    """
    # Create basic configuration
    config = {
        'grid_size': (16, 16, 16),
        'grid_spacing': 0.1,
        'initial_pattern': 'gaussian',
        'history_length': 500,
        'log_level': 'INFO'
    }
    
    # Initialize system
    system = UTCHSSystem(config)
    print(f"Running simulation for {num_ticks} ticks...")
    
    # Run simulation
    states = system.run_simulation(num_ticks)
    print(f"Simulation completed with {len(states)} states recorded")
    
    return system

def compare_linear_vs_nonlinear(position_history: Dict[int, List[Dict]]) -> Dict:
    """
    Compare linear vs. nonlinear meta-pattern detection.
    
    Args:
        position_history: Position history data
        
    Returns:
        Dictionary with comparison results
    """
    # System state for adaptive corrections
    system_state = {
        'global_coherence': 0.85,
        'global_stability': 0.75,
        'energy_level': 0.8,
        'phase_recursion_depth': 3,
        'energy_stability': 0.7,
        'phase_stability': 0.8
    }
    
    # Create detectors
    linear_detector = MetaPatternDetector(config={
        'enable_phase_locking': False,
        'enable_coherence_gating': False,
        'max_recursion_order': 5
    })
    
    nonlinear_detector = MetaPatternDetector(config={
        'enable_phase_locking': True,
        'enable_coherence_gating': True,
        'max_recursion_order': 5
    })
    
    # Detect patterns with both detectors
    linear_results = linear_detector.detect_all_meta_patterns(position_history)
    nonlinear_results = nonlinear_detector.detect_all_meta_patterns(position_history, system_state)
    
    # Compare results
    comparison = {
        'linear': {
            'detected_orders': [order for order, result in linear_results.items() if result.get('detected', False)],
            'meta_cycle_strengths': {order: result.get('meta_cycle_strength', 0) 
                                  for order, result in linear_results.items()}
        },
        'nonlinear': {
            'detected_orders': [order for order, result in nonlinear_results.items() if result.get('detected', False)],
            'meta_cycle_strengths': {order: result.get('meta_cycle_strength', 0) 
                                  for order, result in nonlinear_results.items()}
        }
    }
    
    # Add meta cycle calculations to show differences
    comparison['cycle_comparisons'] = {}
    for order in range(2, 6):
        for position in [3, 6, 9]:
            linear_cycle = linear_detector._calculate_meta_position_cycle(position, order)
            nonlinear_cycle = nonlinear_detector._calculate_meta_position_cycle(position, order, system_state)
            
            comparison['cycle_comparisons'][f"Position {position} at order {order}"] = {
                'linear': linear_cycle,
                'nonlinear': nonlinear_cycle,
                'difference': linear_cycle - nonlinear_cycle
            }
    
    return comparison

def visualize_comparison(comparison: Dict) -> None:
    """
    Visualize the comparison between linear and nonlinear detection.
    
    Args:
        comparison: Comparison results
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot meta cycle strengths
    plt.subplot(2, 1, 1)
    linear_strengths = comparison['linear']['meta_cycle_strengths']
    nonlinear_strengths = comparison['nonlinear']['meta_cycle_strengths']
    
    orders = sorted(set(list(linear_strengths.keys()) + list(nonlinear_strengths.keys())))
    linear_values = [linear_strengths.get(order, 0) for order in orders]
    nonlinear_values = [nonlinear_strengths.get(order, 0) for order in orders]
    
    width = 0.35
    x = np.arange(len(orders))
    plt.bar(x - width/2, linear_values, width, label='Linear')
    plt.bar(x + width/2, nonlinear_values, width, label='Nonlinear')
    
    plt.xlabel('Recursion Order')
    plt.ylabel('Meta-Cycle Strength')
    plt.title('Meta-Pattern Strength Comparison')
    plt.xticks(x, [str(order) for order in orders])
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot cycle comparisons
    plt.subplot(2, 1, 2)
    cycle_data = comparison['cycle_comparisons']
    labels = []
    linear_cycles = []
    nonlinear_cycles = []
    
    for label, data in cycle_data.items():
        labels.append(label)
        linear_cycles.append(data['linear'])
        nonlinear_cycles.append(data['nonlinear'])
    
    x = np.arange(len(labels))
    plt.bar(x - width/2, linear_cycles, width, label='Linear')
    plt.bar(x + width/2, nonlinear_cycles, width, label='Nonlinear')
    
    plt.xlabel('Position and Recursion Order')
    plt.ylabel('Meta-Cycle Value')
    plt.title('Meta-Cycle Calculation Comparison')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('nonlinear_recursion_comparison.png')
    print("Visualization saved to 'nonlinear_recursion_comparison.png'")

def demonstrate_phase_locking() -> None:
    """Demonstrate torsional phase-locking between recursion levels."""
    # Create test position data
    position_data = {
        1: {
            3: {'phase': 0.0},
            6: {'phase': np.pi/2},
            9: {'phase': np.pi}
        },
        2: {
            3: {'phase': np.pi/4},
            6: {'phase': 3*np.pi/4},
            9: {'phase': 5*np.pi/4}
        }
    }
    
    # Create phase-locker with different strengths
    phase_lockers = {
        'weak': TorsionalPhaseLock(config={'lock_strength': 0.3}),
        'medium': TorsionalPhaseLock(config={'lock_strength': 0.6}),
        'strong': TorsionalPhaseLock(config={'lock_strength': 0.9})
    }
    
    results = {}
    
    # Apply phase-locking with different strengths
    for name, locker in phase_lockers.items():
        # Create a deep copy of the position data
        data_copy = {
            level: {
                pos: {'phase': phase} for pos, phase in positions.items()
            } for level, positions in position_data.items()
        }
        
        # Apply phase-locking
        aligned_data = locker.align_recursion_levels(data_copy, 1, 2)
        
        # Calculate alignment metrics
        alignment_metrics = {}
        for pos in [3, 6, 9]:
            original_phase = position_data[2][pos]['phase']
            aligned_phase = aligned_data[2][pos]['phase']
            
            # Calculate angle difference
            diff = np.abs(np.angle(np.exp(1j * aligned_phase) / np.exp(1j * original_phase)))
            
            alignment_metrics[f"Position {pos}"] = {
                'original_phase': float(original_phase),
                'aligned_phase': float(aligned_phase),
                'phase_shift': float(diff)
            }
        
        results[name] = alignment_metrics
    
    # Print results
    print("\nTorsional Phase-Locking Demonstration:")
    print(json.dumps(results, indent=2))

def demonstrate_coherence_gating() -> None:
    """Demonstrate coherence gating for pattern filtering."""
    # Create test patterns with varying coherence
    patterns = [
        {
            'name': 'Highly Coherent',
            'phase_coherence': 0.9,
            'energy_coherence': 0.85,
            'temporal_coherence': 0.95
        },
        {
            'name': 'Medium Coherence',
            'phase_coherence': 0.7,
            'energy_coherence': 0.65,
            'temporal_coherence': 0.75
        },
        {
            'name': 'Low Coherence',
            'phase_coherence': 0.5,
            'energy_coherence': 0.45,
            'temporal_coherence': 0.55
        },
        {
            'name': 'Very Low Coherence',
            'phase_coherence': 0.3,
            'energy_coherence': 0.25,
            'temporal_coherence': 0.35
        }
    ]
    
    # Create coherence gates with different thresholds
    gates = {
        'strict': CoherenceGate(config={'base_threshold': 0.8, 'recursion_factor': 0.05}),
        'moderate': CoherenceGate(config={'base_threshold': 0.6, 'recursion_factor': 0.1}),
        'lenient': CoherenceGate(config={'base_threshold': 0.4, 'recursion_factor': 0.15})
    }
    
    results = {}
    
    # Test each gate at different recursion orders
    for name, gate in gates.items():
        gate_results = {}
        
        for order in [1, 3, 5]:
            pattern_results = {}
            
            for pattern in patterns:
                is_coherent = gate.is_coherent(pattern, order)
                threshold = gate.calculate_threshold(order)
                
                pattern_results[pattern['name']] = {
                    'coherent': is_coherent,
                    'threshold': float(threshold),
                    'combined_score': float(gate.calculate_combined_coherence(
                        pattern['phase_coherence'],
                        pattern['energy_coherence'],
                        pattern['temporal_coherence']
                    ))
                }
            
            gate_results[f"Order {order}"] = pattern_results
        
        results[name] = gate_results
    
    # Print results
    print("\nCoherence Gating Demonstration:")
    print(json.dumps(results, indent=2))

def demonstrate_mobius_correction() -> None:
    """Demonstrate Möbius correction for higher recursion orders."""
    # Create a simple test grid
    grid_size = 10
    x, y = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))
    z = x + 1j * y
    
    # Apply Möbius transformations with different recursion orders
    results = {}
    
    for order in [1, 3, 5]:
        # Create Möbius transformation for this order
        mobius = MobiusTransformation(
            complex(1.0, 0.1),  # a
            complex(0.1, 0.0),  # b
            complex(0.0, 0.1),  # c
            complex(1.0, -0.1), # d
            recursion_order=order
        )
        
        # Get corrected parameters
        a, b, c, d = mobius.get_corrected_parameters()
        
        # Transform the grid
        transformed = mobius.transform(z.flatten()).reshape(z.shape)
        
        # Calculate correction metrics
        results[f"Order {order}"] = {
            'parameters': {
                'a': f"{a.real:.4f} + {a.imag:.4f}j",
                'b': f"{b.real:.4f} + {b.imag:.4f}j",
                'c': f"{c.real:.4f} + {c.imag:.4f}j",
                'd': f"{d.real:.4f} + {d.imag:.4f}j"
            },
            'determinant': f"{(a*d - b*c).real:.4f} + {(a*d - b*c).imag:.4f}j",
            'max_displacement': float(np.max(np.abs(transformed - z)))
        }
    
    # Print results
    print("\nMöbius Correction Demonstration:")
    print(json.dumps(results, indent=2))

def main():
    """Run the example demonstrations."""
    # Create RecursionTracker instance
    recursion_tracker = RecursionTracker.get_instance()
    recursion_tracker.reset()
    
    # Method 1: Run a full simulation
    if False:  # Set to True to run full simulation (can be time-consuming)
        system = simulate_system(num_ticks=1000)
        position_history = recursion_tracker.get_position_history()
    else:
        # Method 2: Use synthetic position history for demonstration
        position_history = create_synthetic_position_history()
    
    # Compare linear vs. nonlinear detection
    comparison = compare_linear_vs_nonlinear(position_history)
    
    # Print comparison summary
    print("\nComparison Summary:")
    print(f"Linear detection found patterns at orders: {comparison['linear']['detected_orders']}")
    print(f"Nonlinear detection found patterns at orders: {comparison['nonlinear']['detected_orders']}")
    
    # Visualize comparison
    visualize_comparison(comparison)
    
    # Demonstrate other components
    demonstrate_phase_locking()
    demonstrate_coherence_gating()
    demonstrate_mobius_correction()

def create_synthetic_position_history() -> Dict[int, List[Dict]]:
    """
    Create synthetic position history for demonstration purposes.
    
    Returns:
        Dictionary with position history data
    """
    print("Creating synthetic position history data...")
    
    # Create position history with predictable patterns
    history = {}
    
    # Create positions for depth 1 (base level)
    history[1] = []
    for cycle in range(1, 14):
        for tick in range(10 * cycle, 10 * cycle + 10):
            position = {
                'position_number': cycle % 13 + 1,
                'phase': (cycle / 13) * 2 * np.pi + np.random.normal(0, 0.1),
                'energy_level': cycle * 0.1 + np.random.normal(0, 0.05),
                'absolute_position': {'cycle': cycle},
                'tick': tick,
                'spatial_location': (
                    np.cos(cycle / 13 * 2 * np.pi), 
                    np.sin(cycle / 13 * 2 * np.pi), 
                    0.0
                )
            }
            history[1].append(position)
    
    # Create positions for depth 2 (meta level 1)
    history[2] = []
    for cycle in [3, 6, 9, 12, 18, 24]:
        for tick in range(100 + 20 * cycle, 100 + 20 * cycle + 20):
            position = {
                'position_number': (cycle // 3) % 13 + 1,
                'phase': (cycle / 24) * 2 * np.pi + np.random.normal(0, 0.15),
                'energy_level': cycle * 0.2 + np.random.normal(0, 0.1),
                'absolute_position': {'cycle': cycle},
                'tick': tick,
                'spatial_location': (
                    0.5 * np.cos(cycle / 24 * 2 * np.pi), 
                    0.5 * np.sin(cycle / 24 * 2 * np.pi), 
                    0.1
                )
            }
            history[2].append(position)
    
    # Create positions for depth 3 (meta level 2)
    history[3] = []
    for cycle in [12, 24, 36, 48]:
        for tick in range(1000 + 50 * cycle, 1000 + 50 * cycle + 30):
            position = {
                'position_number': (cycle // 12) % 13 + 1,
                'phase': (cycle / 48) * 2 * np.pi + np.random.normal(0, 0.2),
                'energy_level': cycle * 0.3 + np.random.normal(0, 0.15),
                'absolute_position': {'cycle': cycle},
                'tick': tick,
                'spatial_location': (
                    0.25 * np.cos(cycle / 48 * 2 * np.pi), 
                    0.25 * np.sin(cycle / 48 * 2 * np.pi), 
                    0.2
                )
            }
            history[3].append(position)
    
    print(f"Created synthetic history with {len(history)} recursion depths:")
    for depth, positions in history.items():
        print(f"  Depth {depth}: {len(positions)} positions")
    
    return history

if __name__ == "__main__":
    main() 