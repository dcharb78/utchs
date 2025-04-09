#!/usr/bin/env python3
"""
Test script to verify RecursionTracker singleton usage across multiple modules.

This script demonstrates that the RecursionTracker singleton pattern works
correctly when accessed from different modules, ensuring a single instance
is shared throughout the application.
"""

import os
import sys
import numpy as np
from typing import Dict, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Create first module to use RecursionTracker
def module1_initialize_tracker():
    """First module that initializes the RecursionTracker."""
    print("\n=== Module 1: Initializing RecursionTracker ===")
    from utchs.core.recursion_tracker import RecursionTracker
    
    # Initialize with specific configuration
    config = {
        'recursion_history_length': 500,
        'max_recursion_depth': 5
    }
    
    tracker = RecursionTracker.get_instance(config)
    print(f"Module 1 - tracker ID: {id(tracker)}")
    print(f"Module 1 - max_recursion_depth: {tracker.max_recursion_depth}")
    print(f"Module 1 - history_length: {tracker.max_history_length}")
    
    # Add some test data
    tracker.reset()
    
    # Add dummy transition data
    test_transition = {
        'tick': 100,
        'recursion_depth': 1,
        'phase_shift': 0.5,
        'energy_ratio': 1.2,
        'phi_phase_resonance': True,
        'phi_energy_resonance': False,
        'position': {'position_number': 10, 'phase': 2.0},
        'previous_position': {'position_number': 10, 'phase': 1.5}
    }
    
    tracker.recursion_transitions.append(test_transition)
    print(f"Module 1 - Added test transition, count: {len(tracker.recursion_transitions)}")
    
    return tracker


# Create second module to use RecursionTracker
def module2_access_tracker():
    """Second module that accesses the RecursionTracker without initializing it."""
    print("\n=== Module 2: Accessing RecursionTracker ===")
    from utchs.core.recursion_tracker import RecursionTracker
    
    # Access existing instance
    tracker = RecursionTracker.get_instance()
    print(f"Module 2 - tracker ID: {id(tracker)}")
    print(f"Module 2 - max_recursion_depth: {tracker.max_recursion_depth}")
    print(f"Module 2 - history_length: {tracker.max_history_length}")
    
    # Check that we can access data added by Module 1
    print(f"Module 2 - Transitions count: {len(tracker.recursion_transitions)}")
    if tracker.recursion_transitions:
        transition = tracker.recursion_transitions[0]
        print(f"Module 2 - First transition tick: {transition['tick']}")
        print(f"Module 2 - Phi phase resonance: {transition['phi_phase_resonance']}")
    
    # Add another test transition
    test_transition = {
        'tick': 200,
        'recursion_depth': 2,
        'phase_shift': -0.3,
        'energy_ratio': 1.6,
        'phi_phase_resonance': False,
        'phi_energy_resonance': True,
        'position': {'position_number': 10, 'phase': 3.0},
        'previous_position': {'position_number': 10, 'phase': 3.3}
    }
    
    tracker.recursion_transitions.append(test_transition)
    print(f"Module 2 - Added second transition, count: {len(tracker.recursion_transitions)}")
    
    return tracker


# Create third module that tries to modify configuration
def module3_modify_config():
    """Third module that tries to modify the configuration of the singleton."""
    print("\n=== Module 3: Attempting to Modify RecursionTracker Config ===")
    from utchs.core.recursion_tracker import RecursionTracker
    
    # Try to create with different config
    new_config = {
        'recursion_history_length': 200,
        'max_recursion_depth': 10
    }
    
    tracker = RecursionTracker.get_instance(new_config)
    print(f"Module 3 - tracker ID: {id(tracker)}")
    print(f"Module 3 - max_recursion_depth: {tracker.max_recursion_depth}")
    print(f"Module 3 - history_length: {tracker.max_history_length}")
    
    # Check if we can see the transitions added by modules 1 and 2
    print(f"Module 3 - Transitions count: {len(tracker.recursion_transitions)}")
    for i, transition in enumerate(tracker.recursion_transitions):
        print(f"Module 3 - Transition {i+1} tick: {transition['tick']}")
    
    return tracker


def main():
    """Run tests to verify singleton pattern across modules."""
    print("=== Testing RecursionTracker Singleton Across Modules ===")
    
    # Run each module function to simulate separate modules using the tracker
    tracker1 = module1_initialize_tracker()
    tracker2 = module2_access_tracker()
    tracker3 = module3_modify_config()
    
    # Verify all modules have the same instance
    print("\n=== Verification ===")
    print(f"All three modules have the same tracker instance: {tracker1 is tracker2 is tracker3}")
    
    # Verify the config wasn't changed by subsequent calls
    print(f"Final tracker max_recursion_depth: {tracker1.max_recursion_depth}")
    print(f"Final tracker history_length: {tracker1.max_history_length}")
    
    # Verify all transitions are visible
    print(f"Final transitions count: {len(tracker1.recursion_transitions)}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main() 