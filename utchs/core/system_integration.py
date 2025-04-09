"""
System integration module for the UTCHS framework.

This module implements the SystemIntegrator class, which integrates
recursion tracking with improved data sufficiency validation and
tiered calculation approach to address performance and stability issues.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import matplotlib.pyplot as plt
from collections import deque

from ..utils.logging_config import get_logger
from ..core.recursion_tracker import RecursionTracker
from ..core.transition_analyzer import TransitionAnalyzer
from ..core.fractal_analyzer import FractalAnalyzer
from ..visualization.recursion_vis import RecursionVisualizer
from .meta_pattern_detector import MetaPatternDetector

logger = get_logger(__name__)

class SystemIntegrator:
    """
    Integrates recursion tracking with improved data validation and performance.
    
    This class enhances the recursion tracking integration with:
    1. Data sufficiency validation to prevent calculations with insufficient data
    2. Tiered calculation approach for improved performance
    3. Enhanced error handling to prevent division by zero errors
    4. Deferred calculation of expensive metrics
    """
    
    def __init__(self, system, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the system integrator.
        
        Args:
            system: UTCHSSystem instance
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.system = system
        
        # Create recursion tracker using singleton pattern
        self.recursion_tracker = RecursionTracker.get_instance(self.config)
        
        # Create analyzers
        self.transition_analyzer = TransitionAnalyzer(self.recursion_tracker, self.config)
        self.fractal_analyzer = FractalAnalyzer(self.recursion_tracker, self.config)
        
        # Create visualizer
        self.visualizer = RecursionVisualizer(self.recursion_tracker, self.config)
        
        # Initialize meta-pattern detector
        self.meta_pattern_detector = MetaPatternDetector(self.config.get('meta_pattern', {}))
        
        # Output directory for visualizations
        self.output_dir = self.config.get('output_dir', 'recursion_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Analysis interval (in ticks)
        self.analysis_interval = self.config.get('analysis_interval', 100)
        
        # Visualization interval (in ticks)
        self.visualization_interval = self.config.get('visualization_interval', 500)
        
        # Data sufficiency thresholds (configurable)
        self.min_ticks = self.config.get('min_ticks', 50)
        self.min_positions = self.config.get('min_positions', 20)
        self.min_transitions = self.config.get('min_transitions', 5)
        self.min_depths = self.config.get('min_depths', 1)
        
        # Results cache
        self.metrics_cache = {}
        self.last_cache_tick = 0
        self.cache_validity_ticks = self.config.get('cache_validity_ticks', 10)
        
        # Last analyzed tick
        self.last_analyzed_tick = 0
        
        # Last visualized tick
        self.last_visualized_tick = 0
        
        logger.info("SystemIntegrator initialized with data sufficiency validation")
    
    def _has_sufficient_data_for_analysis(self):
        """
        Check if we have enough data for advanced analytical metrics.
        
        This ensures we don't attempt complex calculations until we have
        meaningful data to work with.
        
        Returns:
            Boolean indicating if we have sufficient data
        """
        # Check current tick count
        if self.system.current_tick < self.min_ticks:
            logger.debug(f"Insufficient ticks ({self.system.current_tick}/{self.min_ticks}) for advanced analysis")
            return False
        
        # Check position history
        total_positions = sum(len(history) for history in self.recursion_tracker.position_history.values())
        if total_positions < self.min_positions:
            logger.debug(f"Insufficient positions ({total_positions}/{self.min_positions}) for advanced analysis")
            return False
            
        # Check transition history
        transitions = self.recursion_tracker.get_recursion_transitions()
        if len(transitions) < self.min_transitions:
            logger.debug(f"Insufficient transitions ({len(transitions)}/{self.min_transitions}) for advanced analysis")
            return False
            
        # Check recursion depths
        if len(self.recursion_tracker.position_history) < self.min_depths:
            logger.debug(f"Insufficient recursion depths ({len(self.recursion_tracker.position_history)}/{self.min_depths})")
            return False
        
        # We have enough data
        logger.debug(f"Sufficient data for advanced analysis at tick {self.system.current_tick}")
        return True
    
    def update(self, tick: int) -> None:
        """
        Update recursion tracking on system tick.
        
        This method should be called on each system tick to track positions
        and detect transitions.
        
        Args:
            tick: Current system tick
        """
        # Track positions across all tori
        self._track_positions(tick)
        
        # Run analysis at intervals
        if tick % self.analysis_interval == 0:
            self._run_analysis(tick)
        
        # Run visualization at intervals
        if tick % self.visualization_interval == 0:
            self._run_visualization(tick)
    
    def _track_positions(self, tick: int) -> None:
        """
        Track positions across all tori.
        
        Args:
            tick: Current system tick
        """
        recursion_depth = self.system.phase_recursion_depth
        
        for torus in self.system.tori:
            for structure in torus.structures:
                for cycle in structure.cycles:
                    for position in cycle.positions:
                        self.recursion_tracker.track_position(position, recursion_depth, tick)
    
    def _get_recursion_metrics(self):
        """
        Get current recursion metrics with tiered calculation approach.
        
        Returns:
            Dictionary with recursion metrics
        """
        # Check cache first
        current_tick = self.system.current_tick
        if (current_tick - self.last_cache_tick <= self.cache_validity_ticks and 
            self.metrics_cache):
            logger.debug(f"Using cached metrics from tick {self.last_cache_tick}")
            return self.metrics_cache.copy()
        
        try:
            # Initialize with default metrics
            metrics = {
                'transitions_count': 0,
                'phi_resonance_count': 0,
                'phi_resonance_percentage': 0.0,
                'self_similarity': None,
                'fractal_dimension': None,
                'meta_pattern_detected': False,
                'meta_pattern_strength': 0.0,
                'has_sufficient_data': False
            }
            
            # Get current recursion transitions
            transitions = self.recursion_tracker.get_recursion_transitions()
            metrics['transitions_count'] = len(transitions)
            
            # --- Basic Metrics (always calculate) ---
            if transitions:
                try:
                    # Count phi resonances
                    phi_resonance_count = sum(
                        1 for t in transitions 
                        if t.get('phi_phase_resonance') or t.get('phi_energy_resonance')
                    )
                    metrics['phi_resonance_count'] = phi_resonance_count
                    
                    # Safe division
                    if len(transitions) > 0:
                        metrics['phi_resonance_percentage'] = phi_resonance_count / len(transitions)
                    else:
                        metrics['phi_resonance_percentage'] = 0.0
                except Exception as e:
                    logger.warning(f"Error calculating basic phi resonance metrics: {str(e)}")
            
            # --- Advanced Metrics (only with sufficient data) ---
            has_sufficient_data = self._has_sufficient_data_for_analysis()
            metrics['has_sufficient_data'] = has_sufficient_data
            
            if has_sufficient_data:
                # Only calculate these on analysis_interval ticks or when explicitly requested
                if self.system.current_tick % self.analysis_interval == 0:
                    try:
                        # Get fractal metrics
                        self_similarity = self.fractal_analyzer.calculate_self_similarity()
                        if self_similarity:
                            metrics['self_similarity'] = self_similarity.get('self_similarity')
                    except Exception as e:
                        logger.warning(f"Error calculating self-similarity: {str(e)}")
                    
                    try:
                        # Calculate fractal dimension
                        fractal_dimension = self.fractal_analyzer.calculate_fractal_dimension()
                        if fractal_dimension:
                            metrics['fractal_dimension'] = fractal_dimension.get('dimension')
                    except Exception as e:
                        logger.warning(f"Error calculating fractal dimension: {str(e)}")
                    
                    try:
                        # Get meta-pattern metrics
                        meta_pattern = self.meta_pattern_detector.detect_meta_patterns(
                            self.recursion_tracker.position_history
                        )
                        if meta_pattern:
                            metrics['meta_pattern_detected'] = meta_pattern.get('detected', False)
                            metrics['meta_pattern_strength'] = meta_pattern.get('meta_cycle_strength', 0)
                    except Exception as e:
                        logger.warning(f"Error detecting meta-patterns: {str(e)}")
            
            # Update cache
            self.metrics_cache = metrics.copy()
            self.last_cache_tick = current_tick
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting recursion metrics: {str(e)}")
            return {}
    
    def _run_analysis(self, tick: int) -> None:
        """
        Run analysis on recursion data at specified intervals.
        
        Args:
            tick: Current system tick
        """
        logger.info(f"Running recursion analysis at tick {tick}")
        
        # Check for sufficient data
        has_sufficient_data = self._has_sufficient_data_for_analysis()
        if not has_sufficient_data:
            logger.info("Skipping advanced analysis due to insufficient data")
            return
        
        try:
            # Analyze P13 seventh cycle transformation
            p13_analysis = self.transition_analyzer.analyze_p13_seventh_cycle_transformation()
            
            # Analyze octave transitions
            octave_analysis = self.transition_analyzer.analyze_octave_transitions()
            
            # Detect phi resonances
            phi_resonance = self.transition_analyzer.detect_phi_resonances()
            
            # Store analysis timestamp
            self.last_analyzed_tick = tick
            
            # Analyze more complex properties with explicit try/except for each
            self._run_complex_analysis(tick)
        except Exception as e:
            logger.error(f"Error running basic analysis: {str(e)}")
    
    def _run_complex_analysis(self, tick: int) -> None:
        """
        Run computationally expensive analyses with explicit error handling for each.
        
        Args:
            tick: Current system tick
        """
        # Analyze fractal properties - with explicit try/except for each
        try:
            fractal_dimension = self.fractal_analyzer.calculate_fractal_dimension()
            logger.debug(f"Fractal dimension calculated: {fractal_dimension.get('dimension', None)}")
        except Exception as e:
            logger.warning(f"Error calculating fractal dimension: {str(e)}")
            fractal_dimension = None
            
        try:
            multi_scale_entropy = self.fractal_analyzer.calculate_multi_scale_entropy()
        except Exception as e:
            logger.warning(f"Error calculating multi-scale entropy: {str(e)}")
            multi_scale_entropy = None
            
        try:
            self_similarity = self.fractal_analyzer.calculate_self_similarity()
            logger.debug(f"Self-similarity calculated: {self_similarity.get('self_similarity', None)}")
        except Exception as e:
            logger.warning(f"Error calculating self-similarity: {str(e)}")
            self_similarity = None
        
        # Analyze meta-patterns
        try:
            meta_pattern_analysis = self.meta_pattern_detector.detect_meta_patterns(
                self.recursion_tracker.position_history,
                system_state=self._get_system_state()
            )
            
            # Log significant findings
            if meta_pattern_analysis and meta_pattern_analysis.get('detected', False):
                logger.info(f"Meta-pattern detected at tick {tick} with strength {meta_pattern_analysis.get('meta_cycle_strength', 0):.4f}")
        except Exception as e:
            logger.warning(f"Error detecting meta-patterns: {str(e)}")
            meta_pattern_analysis = {'detected': False}
    
    def _get_system_state(self) -> Dict[str, Any]:
        """
        Get current system state for analysis.
        
        Returns:
            Dictionary with system state information
        """
        return {
            'tick': self.system.current_tick,
            'recursion_depth': self.system.phase_recursion_depth,
            'tori_count': len(self.system.tori)
        }
    
    def _run_visualization(self, tick: int) -> None:
        """
        Run visualization of recursion data.
        
        Args:
            tick: Current system tick
        """
        # Check for sufficient data
        has_sufficient_data = self._has_sufficient_data_for_analysis()
        if not has_sufficient_data:
            logger.info("Skipping visualization due to insufficient data")
            return
        
        logger.info(f"Generating recursion visualizations at tick {tick}")
        
        try:
            # Create output directory for this tick
            tick_dir = os.path.join(self.output_dir, f"tick_{tick}")
            os.makedirs(tick_dir, exist_ok=True)
            
            # Visualize Position 10 (recursive seed point) across depths
            p10_vis = self.visualizer.visualize_position_across_depths(
                10, tick, save=True, 
                filename=os.path.join(tick_dir, f"position_10_depths_tick_{tick}.png")
            )
            plt.close(p10_vis)
            
            # Visualize Position 13 (cycle completion point) across depths
            p13_vis = self.visualizer.visualize_position_across_depths(
                13, tick, save=True, 
                filename=os.path.join(tick_dir, f"position_13_depths_tick_{tick}.png")
            )
            plt.close(p13_vis)
            
            # Visualize phi scaling
            phi_vis = self.visualizer.visualize_phi_scaling(
                self.fractal_analyzer, save=True,
                filename=os.path.join(tick_dir, f"phi_scaling_tick_{tick}.png")
            )
            plt.close(phi_vis)
            
            # Store visualization timestamp
            self.last_visualized_tick = tick
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    def generate_report(self, output_file: str = "recursion_analysis_report.txt") -> None:
        """
        Generate a comprehensive report of recursion analysis.
        
        Args:
            output_file: Output file path
        """
        logger.info(f"Generating recursion analysis report to {output_file}")
        
        # Check for sufficient data
        has_sufficient_data = self._has_sufficient_data_for_analysis()
        if not has_sufficient_data:
            logger.warning("Generating limited report due to insufficient data")
        
        with open(output_file, 'w') as f:
            f.write("=======================================================\n")
            f.write("UTCHS RECURSIVE TRANSITION ANALYSIS REPORT\n")
            f.write("=======================================================\n\n")
            
            # System information
            f.write(f"System Tick: {self.system.current_tick}\n")
            f.write(f"Recursion Depth: {self.system.phase_recursion_depth}\n")
            f.write(f"Number of Tori: {len(self.system.tori)}\n")
            f.write(f"Data Sufficiency: {'Yes' if has_sufficient_data else 'No'}\n\n")
            
            if not has_sufficient_data:
                f.write("NOTE: Insufficient data for complete analysis.\n")
                f.write(f"Current tick: {self.system.current_tick}, Required: {self.min_ticks}\n")
                f.write("Limited analysis results are shown below.\n\n")
            
            # Get metrics
            metrics = self._get_recursion_metrics()
            
            # Basic transition information (always available)
            f.write("-------------------------------------------------------\n")
            f.write("BASIC TRANSITION METRICS\n")
            f.write("-------------------------------------------------------\n")
            f.write(f"Total Transitions: {metrics.get('transitions_count', 0)}\n")
            f.write(f"Phi Resonances: {metrics.get('phi_resonance_count', 0)}\n")
            f.write(f"Phi Resonance %: {metrics.get('phi_resonance_percentage', 0)*100:.1f}%\n\n")
            
            # Only include advanced metrics if we have sufficient data
            if has_sufficient_data:
                # P13 Seventh Cycle Transformation
                f.write("-------------------------------------------------------\n")
                f.write("P13 SEVENTH CYCLE TRANSFORMATION ANALYSIS\n")
                f.write("-------------------------------------------------------\n")
                
                p13_analysis = self.transition_analyzer.detected_transitions.get('p13_seventh_cycle', [])
                
                if not p13_analysis:
                    f.write("No P13 transformations detected yet.\n\n")
                else:
                    f.write(f"Detected {len(p13_analysis)} P13 transformations.\n\n")
                    
                    # Most recent transformation
                    recent = p13_analysis[-1]
                    f.write("Most recent transformation:\n")
                    f.write(f"  Tick: {recent.get('from_tick', 'unknown')} -> {recent.get('to_tick', 'unknown')}\n")
                    f.write(f"  Phase Shift: {recent.get('phase_shift', 0):.4f}\n")
                    f.write(f"  Energy Ratio: {recent.get('energy_ratio', 0):.4f}\n")
                    
                    phi_phase = recent.get('phi_phase_resonance', False)
                    phi_energy = recent.get('phi_energy_resonance', False)
                    f.write(f"  Phi Resonance: {'Yes' if phi_phase or phi_energy else 'No'}\n\n")
                
                # Fractal Analysis
                f.write("-------------------------------------------------------\n")
                f.write("FRACTAL ANALYSIS\n")
                f.write("-------------------------------------------------------\n")
                
                f.write(f"Fractal Dimension: {metrics.get('fractal_dimension', 'Not calculated')}\n")
                f.write(f"Self-Similarity: {metrics.get('self_similarity', 'Not calculated')}\n\n")
                
                # Meta-Pattern Analysis
                f.write("-------------------------------------------------------\n")
                f.write("META-PATTERN ANALYSIS\n")
                f.write("-------------------------------------------------------\n")
                
                f.write(f"Meta-pattern detected: {'Yes' if metrics.get('meta_pattern_detected', False) else 'No'}\n")
                f.write(f"Meta-cycle strength: {metrics.get('meta_pattern_strength', 0):.4f}\n\n")
            
            # Visualization Paths
            f.write("-------------------------------------------------------\n")
            f.write("VISUALIZATION PATHS\n")
            f.write("-------------------------------------------------------\n")
            
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write(f"Last Visualization: Tick {self.last_visualized_tick}\n\n")
            
            # Conclusion
            f.write("=======================================================\n")
            f.write("CONCLUSION\n")
            f.write("=======================================================\n\n")
            
            if not has_sufficient_data:
                f.write("Insufficient data for conclusive insights. Continue running the system to\n")
                f.write(f"gather more data (at least {self.min_ticks} ticks required).\n\n")
            elif metrics.get('meta_pattern_detected', False):
                f.write("Meta-pattern detected, confirming the theoretical prediction of recursive 3-6-9 patterns.\n")
                f.write("The system has established a second-order 3-6-9 pattern where cycle 6 becomes a meta-position 3.\n\n")
            else:
                f.write("Analysis shows typical recursion patterns but no clear meta-pattern emergence yet.\n")
                f.write("Continue monitoring for emergence of the recursive 3-6-9 pattern at cycle 6.\n\n")
        
        logger.info(f"Report generated to {output_file}")
    
    def attach_to_system(self) -> None:
        """
        Attach recursion tracking to the UTCHSSystem.
        
        This method modifies the system's advance_tick method to include
        recursion tracking updates.
        """
        original_advance_tick = self.system.advance_tick
        
        def new_advance_tick():
            """Enhanced advance_tick with recursion tracking."""
            result = original_advance_tick()
            self.update(self.system.current_tick)
            return result
        
        self.system.advance_tick = new_advance_tick
        logger.info("Recursion tracking attached to UTCHSSystem")

def integrate_system_tracking(system, config: Optional[Dict[str, Any]] = None) -> SystemIntegrator:
    """
    Integrate system tracking with data sufficiency validation.
    
    This is the main function to use when integrating the system tracker
    with a UTCHS system.
    
    Args:
        system: UTCHSSystem instance
        config: Configuration dictionary (optional)
        
    Returns:
        SystemIntegrator instance
    """
    integrator = SystemIntegrator(system, config)
    integrator.attach_to_system()
    return integrator 