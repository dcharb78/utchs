#!/usr/bin/env python
"""
Example script demonstrating recursion integration with UTCHSSystem.

This script shows how to enhance a UTCHSSystem instance with recursion tracking,
phase-locking, coherence gating, and meta-pattern detection.
"""

import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path to make imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utchs.core.system import UTCHSSystem
from utchs.core.system_integration import integrate_recursion_tracking, create_default_configuration
from utchs.utils.logging_config import configure_logging

def setup_logging(log_level='INFO'):
    """Configure logging for this script."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'recursion_integration_example.log')
    
    # Configure logging
    configure_logging(
        log_level=log_level,
        log_file=log_file,
        console_output=True,
        file_output=True,
        log_dir=log_dir
    )
    
    # Get a logger for this script
    return logging.getLogger('recursion_integration_example')

def create_system(grid_size=(32, 32, 32), grid_spacing=0.25):
    """Create a UTCHSSystem instance with given configuration."""
    config = {
        'grid_size': grid_size,
        'grid_spacing': grid_spacing,
        'history_length': 1000,
        'log_level': 'INFO'
    }
    
    return UTCHSSystem(config)

def enhance_system_with_recursion_tracking(system, output_dir='recursion_output', log_level='INFO'):
    """
    Enhance a UTCHSSystem with recursion tracking.
    
    Args:
        system: UTCHSSystem instance
        output_dir: Directory for output files
        log_level: Logging level for recursion components
        
    Returns:
        UTCHSSystemIntegrator instance
    """
    # Create recursion tracking configuration
    config = create_default_configuration()
    
    # Customize configuration
    config['output_dir'] = output_dir
    config['tracking_interval'] = 1
    config['analysis_interval'] = 50
    config['visualization_interval'] = 100
    config['recursion']['max_history_length'] = 2000
    
    # Integrate recursion tracking
    return integrate_recursion_tracking(system, config)

def run_simulation(system, integrator, num_ticks=1000):
    """
    Run a simulation with enhanced recursion tracking.
    
    Args:
        system: UTCHSSystem instance
        integrator: UTCHSSystemIntegrator instance
        num_ticks: Number of ticks to simulate
        
    Returns:
        Simulation results
    """
    logger.info(f"Running simulation for {num_ticks} ticks")
    
    # Run the simulation
    states = system.run_simulation(num_ticks)
    
    # Generate a report
    report_file = os.path.join(integrator.config['output_dir'], f"final_report_{num_ticks}.txt")
    system.generate_recursion_report(report_file)
    
    logger.info(f"Simulation completed. Report generated: {report_file}")
    
    return states

def analyze_results(system, states):
    """
    Analyze simulation results.
    
    Args:
        system: UTCHSSystem instance
        states: List of state dictionaries from simulation
        
    Returns:
        Analysis results
    """
    logger.info("Analyzing simulation results")
    
    # Get recursion metrics
    recursion_tracker = system.get_recursion_tracker()
    transition_analyzer = system.get_transition_analyzer()
    fractal_analyzer = system.get_fractal_analyzer()
    meta_pattern_detector = system.get_meta_pattern_detector()
    
    # Get transition statistics
    transitions = recursion_tracker.get_recursion_transitions()
    phi_resonances = [t for t in transitions if t.get('phi_phase_resonance') or t.get('phi_energy_resonance')]
    
    # Calculate fractal metrics
    fractal_dimension = fractal_analyzer.calculate_fractal_dimension()
    self_similarity = fractal_analyzer.calculate_self_similarity()
    
    # Get meta-pattern detection results
    meta_pattern = meta_pattern_detector.detect_meta_patterns(
        recursion_tracker.position_history
    )
    
    # Create analysis results
    analysis = {
        'total_transitions': len(transitions),
        'phi_resonances': len(phi_resonances),
        'phi_resonance_percentage': len(phi_resonances) / max(1, len(transitions)),
        'fractal_dimension': fractal_dimension.get('dimension') if fractal_dimension else None,
        'self_similarity': self_similarity.get('self_similarity') if self_similarity else None,
        'meta_pattern': {
            'detected': meta_pattern.get('detected', False),
            'strength': meta_pattern.get('meta_cycle_strength', 0)
        }
    }
    
    # Log analysis results
    logger.info(f"Total transitions: {analysis['total_transitions']}")
    logger.info(f"Phi resonances: {analysis['phi_resonances']} " +
               f"({analysis['phi_resonance_percentage']*100:.1f}%)")
    
    if analysis['fractal_dimension'] is not None:
        logger.info(f"Fractal dimension: {analysis['fractal_dimension']:.4f}")
    
    if analysis['self_similarity'] is not None:
        logger.info(f"Self-similarity: {analysis['self_similarity']:.4f}")
    
    if analysis['meta_pattern']['detected']:
        logger.info(f"Meta-pattern detected with strength: {analysis['meta_pattern']['strength']:.4f}")
    
    return analysis

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='UTCHS Recursion Integration Example')
    parser.add_argument('--ticks', type=int, default=1000, help='Number of ticks to simulate')
    parser.add_argument('--grid-size', type=int, default=32, help='Grid size')
    parser.add_argument('--output-dir', default='recursion_output', help='Output directory')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.log_level)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create system
    system = create_system(grid_size=(args.grid_size, args.grid_size, args.grid_size))
    
    # Enhance system with recursion tracking
    integrator = enhance_system_with_recursion_tracking(system, args.output_dir, args.log_level)
    
    # Run simulation
    states = run_simulation(system, integrator, args.ticks)
    
    # Analyze results
    analysis = analyze_results(system, states)
    
    # Detach components (optional)
    integrator.detach_components()
    
    logger.info("Example completed successfully")
    
    return analysis

if __name__ == '__main__':
    main() 