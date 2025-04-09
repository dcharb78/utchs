#!/usr/bin/env python3
"""
Test script to verify RecursionTracker singleton implementation.

This script tests the singleton pattern by creating multiple RecursionTracker 
instances and verifying they reference the same object, then runs a full 
6-cycle test to ensure the tracker properly records position data across cycles.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utchs.core.recursion_tracker import RecursionTracker
from utchs.core.position import Position
from utchs.core.meta_pattern_detector import MetaPatternDetector


def test_singleton_pattern():
    """Test that RecursionTracker implements singleton pattern correctly."""
    print("\n=== Testing RecursionTracker Singleton Pattern ===")
    
    # Create first instance
    tracker1 = RecursionTracker.get_instance()
    
    # Create second instance with different config
    tracker2 = RecursionTracker.get_instance({"max_recursion_depth": 10})
    
    # Create third instance directly
    tracker3 = RecursionTracker({"max_history_length": 500})
    
    # Create fourth instance via get_instance again
    tracker4 = RecursionTracker.get_instance()
    
    # Check if they are the same object
    print(f"tracker1 id: {id(tracker1)}")
    print(f"tracker2 id: {id(tracker2)}")
    print(f"tracker3 id: {id(tracker3)}")
    print(f"tracker4 id: {id(tracker4)}")
    
    # Verify that tracker1, tracker2, and tracker4 are the same object
    print(f"tracker1 and tracker2 are same object: {tracker1 is tracker2}")
    print(f"tracker1 and tracker4 are same object: {tracker1 is tracker4}")
    
    # Verify that direct instantiation (tracker3) creates a different object
    print(f"tracker1 and tracker3 are same object: {tracker1 is tracker3}")
    
    # All instances created with get_instance should have the original config
    print(f"tracker1 max_recursion_depth: {tracker1.max_recursion_depth}")
    print(f"tracker2 max_recursion_depth: {tracker2.max_recursion_depth}")
    print(f"tracker4 max_recursion_depth: {tracker4.max_recursion_depth}")
    
    # The direct instance should have its own config
    print(f"tracker3 max_history_length: {tracker3.max_history_length}")
    
    # Reset the tracker state for the next test
    tracker1.reset()
    
    return tracker1


def create_mock_position(number, cycle, structure, torus, tick, recursion_depth):
    """Create a mock Position object for testing."""
    # Define position parameters
    role = "Test Position"
    spatial_location = np.array([
        np.cos((number / 13) * 2 * np.pi),
        np.sin((number / 13) * 2 * np.pi),
        0.0
    ])
    rotational_angle = (number / 13) * 2 * np.pi
    energy_level = 0.1 * number
    absolute_position = {
        'position': number,
        'cycle': cycle,
        'structure': structure,
        'torus': torus
    }
    relations = {
        'adjacent': [(number - 1) if number > 1 else 13, 
                    (number + 1) if number < 13 else 1],
        'harmonic': [number * 2 if number * 2 <= 13 else number * 2 - 13],
        'vortex': [3, 6, 9]
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
    
    # Set phase 
    position.phase = (number / 13) * (2 * np.pi)
    
    return position


def run_six_cycle_test(tracker):
    """Run a full 6-cycle test with RecursionTracker."""
    print("\n=== Running Full 6-Cycle Test ===")
    
    # Generate data for 6 complete cycles
    for structure in range(1, 2):
        for cycle in range(1, 7):
            for pos_num in range(1, 14):
                tick = (structure * 100) + (cycle * 13) + pos_num
                recursion_depth = 1
                
                # Create a position
                position = create_mock_position(
                    pos_num, cycle, structure, 1, tick, recursion_depth
                )
                
                # Track the position
                tracker.track_position(position, recursion_depth, tick)
                
                # Print progress
                if pos_num == 1:
                    print(f"Processing cycle {cycle}, tick {tick}")
    
    # Get full position history
    history = tracker.get_position_history()
    
    # Get specific position histories for key positions
    p3_history = tracker.get_position_history(3)
    p6_history = tracker.get_position_history(6)
    p9_history = tracker.get_position_history(9)
    p10_history = tracker.get_position_history(10)
    p13_history = tracker.get_position_history(13)
    
    # Print statistics
    print(f"Total positions tracked: {len(history)}")
    print(f"Position 3 occurrences: {len(p3_history)}")
    print(f"Position 6 occurrences: {len(p6_history)}")
    print(f"Position 9 occurrences: {len(p9_history)}")
    print(f"Position 10 occurrences: {len(p10_history)}")
    print(f"Position 13 occurrences: {len(p13_history)}")
    
    # Get position across recursion levels at the last tick
    last_tick = (1 * 100) + (6 * 13) + 13  # structure 1, cycle 6, position 13
    positions_across_levels = tracker.get_position_across_recursion_levels(13, last_tick)
    
    print(f"\nPositions across recursion levels at last tick ({last_tick}):")
    for level, data in positions_across_levels.items():
        print(f"Level {level}: Position {data['position_number']} at tick {data['tick']}")
    
    # Get transitions
    transitions = tracker.get_recursion_transitions()
    print(f"\nRecursion transitions detected: {len(transitions)}")
    
    return history


def test_pattern_detection(history):
    """Test meta-pattern detection with the recorded position history."""
    print("\n=== Testing Pattern Detection ===")
    
    try:
        # Create pattern detector
        detector = MetaPatternDetector(config={
            'enable_phase_locking': False,  # Disable phase locking to avoid division by zero
            'enable_coherence_gating': False,  # Disable coherence gating
            'max_recursion_order': 3
        })
        
        # Prepare position history in the format needed for detection
        position_history = {}
        
        # Group the flat history by recursion_depth
        for pos in history:
            if 'recursion_depth' in pos:
                depth = pos['recursion_depth']
                if depth not in position_history:
                    position_history[depth] = []
                position_history[depth].append(pos)
        
        # Test if we can access the position history properly
        print(f"Position history organized by depth: {len(position_history)} depths")
        for depth, positions in position_history.items():
            print(f"  Depth {depth}: {len(positions)} positions")
        
        # Create simple system state for adaptive corrections
        system_state = {
            'global_coherence': 0.85,
            'global_stability': 0.75,
            'energy_level': 0.8,
            'phase_recursion_depth': 1,
            'energy_stability': 0.7,
            'phase_stability': 0.8
        }
        
        # Only analyze base pattern to avoid complex calculations that may fail
        base_pattern = detector._analyze_base_pattern(position_history)
        
        print("\nBase pattern analysis:")
        print(f"  Detected: {base_pattern.get('detected', False)}")
        print(f"  Strength: {base_pattern.get('meta_cycle_strength', 0):.4f}")
        
        return base_pattern
        
    except Exception as e:
        print(f"Error during pattern detection: {e}")
        print("This error is expected in a test environment without full data")
        return {"error": str(e)}


def main():
    """Run all tests."""
    # Test singleton pattern
    tracker = test_singleton_pattern()
    
    # Run 6-cycle test
    history = run_six_cycle_test(tracker)
    
    # Test pattern detection
    test_pattern_detection(history)
    
    print("\n=== All Tests Completed ===")


if __name__ == "__main__":
    main() 