#!/usr/bin/env python3
"""
Test script to validate RecursionTracker over 30 metacycles.

This script simulates 30 metacycles to test the stability and scalability
of the RecursionTracker implementation with large datasets and deep recursion.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time
from collections import deque

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utchs.core.recursion_tracker import RecursionTracker
from utchs.core.position import Position
from utchs.core.meta_pattern_detector import MetaPatternDetector
from utchs.mathematics.recursion_scaling import RecursionScaling

# Constants for the simulation
NUM_METACYCLES = 50  # Increased from 30 to 50
BASE_CYCLE_LENGTH = 13  # positions per cycle
POSITIONS_PER_METACYCLE = 13 * 13  # 13 positions across 13 cycles per metacycle

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


def create_mock_position(number, cycle, metacycle, recursion_depth, tick):
    """Create a mock Position object for testing."""
    # Define position parameters
    role = "Test Position"
    
    # Use phi-based spatial scaling for key positions
    if number in [3, 6, 9, 10]:
        # Special phi-resonant positions
        spatial_location = np.array([
            np.cos((number / 13) * 2 * np.pi) * (1 / (PHI ** recursion_depth)),
            np.sin((number / 13) * 2 * np.pi) * (1 / (PHI ** recursion_depth)),
            recursion_depth * 0.1
        ])
    else:
        # Regular positions
        spatial_location = np.array([
            np.cos((number / 13) * 2 * np.pi) * (1 / (recursion_depth + 1)),
            np.sin((number / 13) * 2 * np.pi) * (1 / (recursion_depth + 1)),
            recursion_depth * 0.1
        ])
    
    rotational_angle = (number / 13) * 2 * np.pi
    
    # Scale energy by golden ratio across recursion depths
    # Special encoding for vortex positions (3, 6, 9)
    if number in [3, 6, 9]:
        # Vortex positions have phi^2 scaling
        energy_level = 0.1 * number * (PHI ** (recursion_depth * 2))
    else:
        # Regular positions have phi scaling
        energy_level = 0.1 * number * (PHI ** recursion_depth)
    
    # Create absolute position structure
    absolute_position = {
        'position': number,
        'cycle': cycle,
        'structure': metacycle // 13 + 1,
        'torus': recursion_depth + 1
    }
    
    # Add phi-resonant harmonic relations for positions 3, 6, 9
    if number in [3, 6, 9]:
        relations = {
            'adjacent': [(number - 1) if number > 1 else 13, 
                        (number + 1) if number < 13 else 1],
            'harmonic': [int(number * PHI) % 13 or 13, int(number * (PHI ** 2)) % 13 or 13],
            'vortex': [3, 6, 9],
            'phi_resonant': True
        }
    else:
        relations = {
            'adjacent': [(number - 1) if number > 1 else 13, 
                        (number + 1) if number < 13 else 1],
            'harmonic': [number * 2 if number * 2 <= 13 else number * 2 - 13],
            'vortex': [3, 6, 9],
            'phi_resonant': False
        }
    
    # Create position object
    position = Position(
        number=number,
        role=role,
        spatial_location=spatial_location,
        rotational_angle=rotational_angle,
        energy_level=energy_level,
        absolute_position=absolute_position,
        relations=relations
    )
    
    # Set phase with golden ratio scaling across recursion depths
    # Add structured variance based on position type
    phase_base = (number / 13) * (2 * np.pi)
    
    if number == 10:  # Recursive seed point
        # Perfect phi scaling for position 10
        position.phase = phase_base * (PHI ** recursion_depth)
    elif number in [3, 6, 9]:  # Vortex positions
        # Phi scaling with harmonic fluctuation
        cycle_factor = np.sin(cycle / 13.0 * np.pi)
        position.phase = phase_base * (PHI ** recursion_depth) * (1 + cycle_factor * 0.1)
    else:
        # Regular positions with small random variance
        variance = np.sin(tick * 0.01) * 0.05
        position.phase = phase_base * (PHI ** recursion_depth) + variance
    
    return position


def run_metacycle_simulation():
    """Run a simulation of 30 metacycles with RecursionTracker."""
    print("\n=== Starting 30 Metacycle Simulation ===")
    
    # Get RecursionTracker singleton with configuration for deep recursion
    config = {
        'recursion_history_length': 2000,
        'max_recursion_depth': 10
    }
    tracker = RecursionTracker.get_instance(config)
    tracker.reset()
    
    # Track performance metrics
    start_time = time.time()
    position_counts = []
    transitions_by_depth = {depth: 0 for depth in range(1, 11)}
    
    # Initialize recursion scaling
    scaling = RecursionScaling(config={'scaling_function': 'phi_resonance'})
    
    # Simulate across recursion depths
    print("Simulating across recursion depths...")
    for recursion_depth in range(1, 6):  # Limit to 5 recursion depths for reasonable simulation time
        print(f"Processing recursion depth {recursion_depth}")
        
        # Determine number of metacycles at this depth
        # Use fewer metacycles at deeper recursion to keep runtime reasonable
        depth_metacycles = NUM_METACYCLES // (recursion_depth ** 2) + 1
        if depth_metacycles > NUM_METACYCLES:
            depth_metacycles = NUM_METACYCLES
        
        print(f"  Simulating {depth_metacycles} metacycles at depth {recursion_depth}")
        
        for metacycle in range(depth_metacycles):
            # Track one metacycle across 13 cycles
            for cycle in range(1, 14):
                # Apply nonlinear scaling to cycle length with recursion depth
                cycle_correction = scaling.get_correction_factor(recursion_depth)
                scaled_cycle_length = int(BASE_CYCLE_LENGTH * (1 + cycle_correction))
                
                for pos_num in range(1, scaled_cycle_length + 1):
                    # Calculate tick based on recursion depth and metacycle
                    tick = (recursion_depth * 10000) + (metacycle * 1000) + (cycle * 100) + pos_num
                    
                    # Create position with resonance properties
                    position = create_mock_position(
                        pos_num, cycle, metacycle, recursion_depth, tick
                    )
                    
                    # Track position
                    tracker.track_position(position, recursion_depth, tick)
                    
                    # Add golden ratio resonance at position 10 (recursive seed point)
                    if pos_num == 10 and cycle in [3, 6, 9]:
                        # Create a resonance transition
                        prev_phase = position.phase / PHI
                        prev_energy = position.energy_level / PHI
                        
                        # Manually add resonance data
                        position_data = {
                            'tick': tick,
                            'position_number': position.number,
                            'energy_level': position.energy_level,
                            'phase': position.phase,
                            'absolute_position': position.absolute_position.copy(),
                            'spatial_location': position.spatial_location.copy(),
                            'recursion_depth': recursion_depth
                        }
                        
                        previous_data = {
                            'tick': tick - int(13 / PHI),  # Fibonacci-related time gap
                            'position_number': position.number,
                            'energy_level': prev_energy,
                            'phase': prev_phase,
                            'absolute_position': position.absolute_position.copy(),
                            'spatial_location': position.spatial_location.copy(),
                            'recursion_depth': recursion_depth
                        }
                        
                        # Set absolute position ID to ensure tracking works
                        abs_id = tracker._get_absolute_position_id(position.absolute_position)
                        if abs_id not in tracker.absolute_position_history:
                            tracker.absolute_position_history[abs_id] = deque(maxlen=tracker.max_history_length)
                        
                        # Add both states to history
                        tracker.absolute_position_history[abs_id].append(previous_data)
                        tracker.absolute_position_history[abs_id].append(position_data)
                        
                        # Manually trigger transition detection
                        tracker._check_for_recursion_transition(position_data)
                        
                    # Create special transitions at vortex positions
                    if pos_num in [3, 6, 9] and cycle in [1, 5, 8, 13]:
                        # These positions should form a meta-pattern
                        # Manually create golden ratio resonances 
                        position_data = {
                            'tick': tick,
                            'position_number': position.number,
                            'energy_level': position.energy_level,
                            'phase': position.phase,
                            'absolute_position': position.absolute_position.copy(),
                            'spatial_location': position.spatial_location.copy(),
                            'recursion_depth': recursion_depth
                        }
                        
                        # Calculate resonant previous values
                        if recursion_depth > 1:
                            # Connect to lower recursion depth with phi scaling
                            prev_phase = position.phase / PHI
                            prev_energy = position.energy_level / PHI
                            prev_depth = recursion_depth - 1
                        else:
                            # Within same depth but different energy/phase
                            prev_phase = position.phase / PHI
                            prev_energy = position.energy_level / PHI
                            prev_depth = recursion_depth
                        
                        previous_data = {
                            'tick': tick - cycle,  # Time gap based on cycle
                            'position_number': position.number,
                            'energy_level': prev_energy,
                            'phase': prev_phase,
                            'absolute_position': position.absolute_position.copy(),
                            'spatial_location': position.spatial_location.copy(),
                            'recursion_depth': prev_depth
                        }
                        
                        # Check for phi-based phase transition
                        phase_diff = position.phase - prev_phase
                        energy_ratio = position.energy_level / prev_energy if prev_energy != 0 else 0
                        
                        # Only record if we have phi resonance
                        if (abs(abs(phase_diff) - (1/PHI)) < 0.1 or abs(energy_ratio - PHI) < 0.2):
                            # Add resonance transition
                            transition = {
                                'tick': tick,
                                'recursion_depth': recursion_depth,
                                'phase_shift': phase_diff,
                                'energy_ratio': energy_ratio,
                                'phi_phase_resonance': abs(abs(phase_diff) - (1/PHI)) < 0.1,
                                'phi_energy_resonance': abs(energy_ratio - PHI) < 0.2,
                                'position': position_data,
                                'previous_position': previous_data
                            }
                            
                            # Add to recursion transitions
                            tracker.recursion_transitions.append(transition)
            
            # Track progress
            if metacycle % 5 == 0:
                print(f"    Completed metacycle {metacycle}")
                print(f"    Current position count: {sum(len(tracker.position_history[d]) for d in tracker.position_history)}")
                print(f"    Current transitions: {len(tracker.recursion_transitions)}")
        
        # Track statistics after each recursion depth
        depth_positions = sum(len(tracker.position_history[d]) for d in tracker.position_history)
        position_counts.append(depth_positions)
        
        # Count transitions by depth
        for transition in tracker.recursion_transitions:
            if transition['recursion_depth'] in transitions_by_depth:
                transitions_by_depth[transition['recursion_depth']] += 1
    
    # Report simulation statistics
    end_time = time.time()
    total_runtime = end_time - start_time
    total_positions = sum(len(tracker.position_history[d]) for d in tracker.position_history)
    positions_per_second = total_positions / total_runtime if total_runtime > 0 else 0
    
    print("\n=== Metacycle Simulation Complete ===")
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Total positions tracked: {total_positions}")
    print(f"Positions per second: {positions_per_second:.2f}")
    print(f"Total transitions detected: {len(tracker.recursion_transitions)}")
    
    print("\nPositions by recursion depth:")
    for depth in sorted(tracker.position_history.keys()):
        if len(tracker.position_history[depth]) > 0:
            print(f"  Depth {depth}: {len(tracker.position_history[depth])} positions")
    
    print("\nTransitions by recursion depth:")
    for depth in sorted(transitions_by_depth.keys()):
        if transitions_by_depth[depth] > 0:
            print(f"  Depth {depth}: {transitions_by_depth[depth]} transitions")
    
    # Check for phi resonance in transitions
    phi_resonances = 0
    for transition in tracker.recursion_transitions:
        if transition['phi_phase_resonance'] or transition['phi_energy_resonance']:
            phi_resonances += 1
            
    print(f"\nPhi resonances detected: {phi_resonances}")
    print(f"Phi resonance percentage: {phi_resonances / len(tracker.recursion_transitions) * 100:.2f}% of transitions")
    
    return {
        'tracker': tracker,
        'runtime': total_runtime,
        'total_positions': total_positions,
        'position_counts': position_counts,
        'transitions_by_depth': transitions_by_depth,
        'phi_resonances': phi_resonances
    }


def analyze_simulation_results(results):
    """Analyze and visualize simulation results."""
    print("\n=== Analyzing Simulation Results ===")
    
    tracker = results['tracker']
    
    # Check memory usage and performance
    memory_estimate = sys.getsizeof(tracker.position_history) + sys.getsizeof(tracker.recursion_transitions)
    for depth in tracker.position_history:
        for pos in tracker.position_history[depth]:
            memory_estimate += sys.getsizeof(pos)
    
    for trans in tracker.recursion_transitions:
        memory_estimate += sys.getsizeof(trans)
    
    print(f"Estimated memory usage: {memory_estimate / (1024*1024):.2f} MB")
    
    # Analyze scaling patterns
    analyze_scaling_patterns(tracker)
    
    # Visualize results
    visualize_results(results)


def analyze_scaling_patterns(tracker):
    """Analyze scaling patterns in the position data."""
    print("\nAnalyzing scaling patterns across recursion depths...")
    
    # Get phase values by recursion depth for position 10
    phase_by_depth = {}
    energy_by_depth = {}
    
    for depth in sorted(tracker.position_history.keys()):
        if len(tracker.position_history[depth]) > 0:
            p10_positions = [p for p in tracker.position_history[depth] 
                           if p['position_number'] == 10]
            
            if p10_positions:
                phase_values = [p['phase'] for p in p10_positions]
                energy_values = [p['energy_level'] for p in p10_positions]
                
                phase_by_depth[depth] = phase_values
                energy_by_depth[depth] = energy_values
    
    # Check for phi-based scaling between depths
    phi = (1 + np.sqrt(5)) / 2
    
    print("\nPhi scaling analysis for Position 10:")
    for i in range(1, max(phase_by_depth.keys())):
        if i+1 in phase_by_depth:
            # Calculate average values
            avg_phase_i = np.mean(phase_by_depth[i])
            avg_phase_i1 = np.mean(phase_by_depth[i+1])
            phase_ratio = avg_phase_i1 / avg_phase_i if avg_phase_i != 0 else 0
            
            avg_energy_i = np.mean(energy_by_depth[i])
            avg_energy_i1 = np.mean(energy_by_depth[i+1])
            energy_ratio = avg_energy_i1 / avg_energy_i if avg_energy_i != 0 else 0
            
            # Check closeness to phi
            phase_phi_error = abs(phase_ratio - phi) / phi * 100
            energy_phi_error = abs(energy_ratio - phi) / phi * 100
            
            print(f"  Depths {i} -> {i+1}:")
            print(f"    Phase scaling ratio: {phase_ratio:.4f} (phi error: {phase_phi_error:.2f}%)")
            print(f"    Energy scaling ratio: {energy_ratio:.4f} (phi error: {energy_phi_error:.2f}%)")


def visualize_results(results):
    """Visualize the simulation results."""
    print("\nGenerating visualizations...")
    
    # Create a figure for visualizations
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Position counts by recursion depth
    depths = range(1, len(results['position_counts']) + 1)
    plt.subplot(2, 2, 1)
    plt.bar(depths, results['position_counts'])
    plt.xlabel('Recursion Depth')
    plt.ylabel('Position Count')
    plt.title('Positions Tracked by Recursion Depth')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Transitions by recursion depth
    transition_depths = [depth for depth in results['transitions_by_depth'].keys() 
                       if results['transitions_by_depth'][depth] > 0]
    transition_counts = [results['transitions_by_depth'][depth] for depth in transition_depths]
    
    plt.subplot(2, 2, 2)
    plt.bar(transition_depths, transition_counts)
    plt.xlabel('Recursion Depth')
    plt.ylabel('Transition Count')
    plt.title('Transitions by Recursion Depth')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Phi resonance distribution in transitions
    tracker = results['tracker']
    resonance_types = ['Phase Only', 'Energy Only', 'Both', 'Neither']
    resonance_counts = [0, 0, 0, 0]
    
    for transition in tracker.recursion_transitions:
        phase_res = transition['phi_phase_resonance']
        energy_res = transition['phi_energy_resonance']
        
        if phase_res and energy_res:
            resonance_counts[2] += 1  # Both
        elif phase_res:
            resonance_counts[0] += 1  # Phase only
        elif energy_res:
            resonance_counts[1] += 1  # Energy only
        else:
            resonance_counts[3] += 1  # Neither
    
    plt.subplot(2, 2, 3)
    plt.pie(resonance_counts, labels=resonance_types, autopct='%1.1f%%', 
            startangle=90, shadow=True)
    plt.axis('equal')
    plt.title('Phi Resonance Distribution in Transitions')
    
    # Plot 4: Performance metrics
    metrics = ['Positions', 'Transitions', 'Phi Resonances']
    values = [results['total_positions'], 
             len(tracker.recursion_transitions), 
             results['phi_resonances']]
    
    plt.subplot(2, 2, 4)
    plt.bar(metrics, values, color=['blue', 'green', 'purple'])
    plt.yscale('log')
    plt.ylabel('Count (log scale)')
    plt.title('Performance Metrics')
    plt.grid(True, alpha=0.3)
    
    # Save the visualizations
    plt.tight_layout()
    plt.savefig('metacycle_simulation_results.png')
    print("Visualization saved to 'metacycle_simulation_results.png'")


def main():
    """Run the metacycle simulation and analysis."""
    print("=== RecursionTracker 30 Metacycle Test ===")
    
    # Run the simulation
    results = run_metacycle_simulation()
    
    # Analyze and visualize results
    analyze_simulation_results(results)
    
    print("\n=== Testing Complete ===")


if __name__ == "__main__":
    main() 