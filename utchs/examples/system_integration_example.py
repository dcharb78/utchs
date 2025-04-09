#!/usr/bin/env python
"""
UTCHS System Integration Example

This example demonstrates the improved system integration with data sufficiency
validation and tiered calculation approach. It shows how the system handles
different data volumes and prevents division by zero errors.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utchs.core.system import UTCHSSystem
from utchs.core.system_integration import SystemIntegrator, integrate_system_tracking
from utchs.utils.logging_config import configure_logging, get_logger

# Set up logger
logger = get_logger(__name__)

def create_default_configuration() -> Dict[str, Any]:
    """
    Create default configuration for system integration.
    
    Returns:
        Configuration dictionary
    """
    return {
        'output_dir': 'recursion_output',
        'analysis_interval': 20,
        'visualization_interval': 100,
        'min_ticks': 50,              # Minimum ticks for advanced analysis
        'min_positions': 20,          # Minimum positions tracked
        'min_transitions': 5,         # Minimum transitions recorded
        'min_depths': 1,              # Minimum recursion depths
        'cache_validity_ticks': 10,   # Cache validity period
        'meta_pattern': {
            'correlation_threshold': 0.75,
            'energy_threshold': 0.5
        }
    }

def run_integration_example(config: Dict[str, Any], max_ticks: int = 200, 
                           output_dir: str = 'system_integration_output') -> None:
    """
    Run a demonstration of the UTCHS system integration with data sufficiency validation.
    
    Args:
        config: Configuration dictionary for the UTCHS system
        max_ticks: Maximum number of ticks to run
        output_dir: Output directory for results
    """
    # Initialize the UTCHS system
    logger.info("Initializing UTCHS system")
    system = UTCHSSystem(config)
    
    # Create recursion integrator
    recursion_config = create_default_configuration()
    recursion_config['output_dir'] = output_dir
    
    # Create integrator with data sufficiency validation
    logger.info("Initializing SystemIntegrator with data sufficiency validation")
    system_integrator = SystemIntegrator(system, recursion_config)
    
    # Attach system integration to system
    system_integrator.attach_to_system()
    
    # Run the system for specified number of ticks
    logger.info(f"Running system for {max_ticks} ticks")
    start_time = time.time()
    
    try:
        # Phase 1: Run with insufficient data (10 ticks)
        logger.info("Phase 1: Running with insufficient data (10 ticks)")
        phase1_end = min(10, max_ticks)
        
        for _ in range(phase1_end):
            system.advance_tick()
            
        # Generate report after phase 1
        phase1_report = os.path.join(output_dir, "phase1_report.txt")
        system_integrator.generate_report(phase1_report)
        logger.info(f"Phase 1 report generated: {phase1_report}")
        
        # Phase 2: Run with borderline data (50 ticks)
        if max_ticks > 10:
            logger.info("Phase 2: Running with borderline data (total 50 ticks)")
            phase2_end = min(50, max_ticks)
            
            for _ in range(phase1_end, phase2_end):
                system.advance_tick()
                
                # Log progress every 10 ticks
                if system.current_tick % 10 == 0:
                    elapsed = time.time() - start_time
                    ticks_per_second = system.current_tick / elapsed if elapsed > 0 else 0
                    logger.info(f"Tick {system.current_tick}/{max_ticks} - {ticks_per_second:.2f} ticks/sec")
            
            # Generate report after phase 2
            phase2_report = os.path.join(output_dir, "phase2_report.txt")
            system_integrator.generate_report(phase2_report)
            logger.info(f"Phase 2 report generated: {phase2_report}")
        
        # Phase 3: Run with sufficient data (100+ ticks)
        if max_ticks > 50:
            logger.info("Phase 3: Running with sufficient data (100+ ticks)")
            
            for _ in range(phase2_end, max_ticks):
                system.advance_tick()
                
                # Log progress every 10 ticks
                if system.current_tick % 10 == 0:
                    elapsed = time.time() - start_time
                    ticks_per_second = system.current_tick / elapsed if elapsed > 0 else 0
                    logger.info(f"Tick {system.current_tick}/{max_ticks} - {ticks_per_second:.2f} ticks/sec")
    
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    
    # Generate final report
    logger.info("Generating final analysis report")
    final_report = os.path.join(output_dir, "final_report.txt")
    system_integrator.generate_report(final_report)
    
    # Log completion
    elapsed_time = time.time() - start_time
    logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Final tick: {system.current_tick}")
    logger.info(f"Final recursion depth: {system.phase_recursion_depth}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Final report: {final_report}")

def performance_test(system_config: Dict[str, Any], output_dir: str = 'performance_test_output'):
    """
    Run performance tests comparing the new system integration with different data volumes.
    
    Args:
        system_config: System configuration dictionary
        output_dir: Output directory for results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize system
    system = UTCHSSystem(system_config)
    
    # Create integrator configuration
    integrator_config = create_default_configuration()
    integrator_config['output_dir'] = output_dir
    
    # Create integrator with data sufficiency validation
    system_integrator = SystemIntegrator(system, integrator_config)
    system_integrator.attach_to_system()
    
    # Performance metrics
    performance_data = {
        'ticks': [],
        'time_per_tick': [],
        'memory_usage': [],
        'has_sufficient_data': []
    }
    
    # Run performance test
    logger.info("Running performance test")
    max_ticks = 100
    
    try:
        for tick in range(max_ticks):
            # Measure time for single tick
            start_time = time.time()
            system.advance_tick()
            elapsed = time.time() - start_time
            
            # Record metrics every 5 ticks
            if tick % 5 == 0:
                metrics = system_integrator._get_recursion_metrics()
                
                performance_data['ticks'].append(tick)
                performance_data['time_per_tick'].append(elapsed)
                performance_data['has_sufficient_data'].append(metrics.get('has_sufficient_data', False))
                
                logger.info(f"Tick {tick}: {elapsed*1000:.2f}ms per tick, " +
                           f"Sufficient data: {metrics.get('has_sufficient_data', False)}")
    
    except KeyboardInterrupt:
        logger.info("Performance test interrupted by user")
    
    # Generate performance report
    logger.info("Generating performance report")
    with open(os.path.join(output_dir, "performance_report.txt"), 'w') as f:
        f.write("UTCHS SYSTEM INTEGRATION PERFORMANCE REPORT\n")
        f.write("===========================================\n\n")
        
        f.write("Performance metrics at different data volumes:\n\n")
        
        f.write("| Tick | Time/Tick (ms) | Sufficient Data |\n")
        f.write("|------|---------------|----------------|\n")
        
        for i in range(len(performance_data['ticks'])):
            tick = performance_data['ticks'][i]
            time_ms = performance_data['time_per_tick'][i] * 1000
            has_data = performance_data['has_sufficient_data'][i]
            
            f.write(f"| {tick:4d} | {time_ms:13.2f} | {str(has_data):16s} |\n")
    
    logger.info(f"Performance report generated in {output_dir}/performance_report.txt")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run UTCHS system integration example')
    parser.add_argument('--max-ticks', type=int, default=150,
                        help='Maximum number of ticks to run')
    parser.add_argument('--grid-size', type=int, default=32,
                        help='Grid size for phase and energy fields')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--output-dir', type=str, default='system_integration_output',
                        help='Output directory for results')
    parser.add_argument('--performance-test', action='store_true',
                        help='Run performance test')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    configure_logging(log_level=args.log_level, console_output=True)
    
    # Create system configuration
    system_config = {
        'grid_size': (args.grid_size, args.grid_size, args.grid_size),
        'grid_spacing': 0.1,
        'history_length': 100,
        'log_level': args.log_level
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.performance_test:
        # Run performance test
        performance_test(system_config, args.output_dir)
    else:
        # Run example
        run_integration_example(system_config, args.max_ticks, args.output_dir) 