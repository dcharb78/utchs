"""
System integration module for the UTCHS framework.

This module provides integration of recursion tracking, phase-locking, coherence gating,
and meta-pattern detection with the main UTCHSSystem.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

from ..utils.logging_config import get_logger
from ..core.recursion_tracker import RecursionTracker
from ..core.transition_analyzer import TransitionAnalyzer
from ..core.fractal_analyzer import FractalAnalyzer
from ..core.meta_pattern_detector import MetaPatternDetector
from ..core.phase_lock import torsional_phase_lock
from ..core.coherence_gate import coherence_gate
from ..visualization.recursion_vis import RecursionVisualizer
from .recursion_integration import RecursionIntegrator

logger = get_logger(__name__)

class UTCHSSystemIntegrator:
    """
    Integrates recursion components with the UTCHSSystem.
    
    This class provides a unified interface for attaching recursion tracking,
    phase-locking, coherence gating, and meta-pattern detection to the
    main UTCHSSystem.
    """
    
    def __init__(self, system, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the system integrator.
        
        Args:
            system: UTCHSSystem instance to integrate components with
            config: Configuration dictionary (optional)
        """
        self.system = system
        self.config = config or {}
        
        # Store original methods for detachment
        self.original_methods = {}
        
        # Component configuration
        recursion_config = self.config.get('recursion', {})
        meta_pattern_config = self.config.get('meta_pattern', {})
        phase_lock_config = self.config.get('phase_lock', {})
        coherence_gate_config = self.config.get('coherence_gate', {})
        
        # Create component instances
        self.recursion_tracker = RecursionTracker.get_instance(recursion_config)
        
        # Configure global phase lock and coherence gate
        torsional_phase_lock.lock_strength = phase_lock_config.get('lock_strength', 0.7)
        coherence_gate.base_threshold = coherence_gate_config.get('base_threshold', 0.7)
        
        # Create integrator components
        self.transition_analyzer = TransitionAnalyzer(self.recursion_tracker, recursion_config)
        self.fractal_analyzer = FractalAnalyzer(self.recursion_tracker, recursion_config)
        self.meta_pattern_detector = MetaPatternDetector(meta_pattern_config)
        
        # Create recursion integrator
        self.recursion_integrator = RecursionIntegrator(system, recursion_config)
        
        # Configuration for tracking and analysis
        self.tracking_interval = self.config.get('tracking_interval', 1)
        self.analysis_interval = self.config.get('analysis_interval', 100)
        self.visualization_interval = self.config.get('visualization_interval', 500)
        
        # Track if components are attached
        self.attached = False
        
        logger.info("UTCHSSystemIntegrator initialized")
    
    def attach_components(self):
        """
        Attach recursion components to the UTCHSSystem.
        
        This method hooks into the UTCHSSystem's methods to add recursion
        tracking, phase-locking, coherence gating, and meta-pattern detection.
        """
        if self.attached:
            logger.warning("Components already attached to system")
            return
        
        # Save original methods for detachment
        self.original_methods['advance_tick'] = self.system.advance_tick
        self.original_methods['_record_state'] = self.system._record_state
        self.original_methods['_update_system_metrics'] = self.system._update_system_metrics
        self.original_methods['analyze_system_state'] = self.system.analyze_system_state
        
        # Attach new methods
        self._attach_advance_tick()
        self._attach_record_state()
        self._attach_update_system_metrics()
        self._attach_analyze_system_state()
        
        # Add recursion-specific methods
        self.system.get_recursion_tracker = self._get_recursion_tracker
        self.system.get_transition_analyzer = self._get_transition_analyzer
        self.system.get_fractal_analyzer = self._get_fractal_analyzer
        self.system.get_meta_pattern_detector = self._get_meta_pattern_detector
        self.system.generate_recursion_report = self._generate_recursion_report
        
        self.attached = True
        logger.info("Recursion components attached to UTCHSSystem")
    
    def detach_components(self):
        """
        Detach recursion components from the UTCHSSystem.
        
        This method restores the original methods of the UTCHSSystem.
        """
        if not self.attached:
            logger.warning("No components attached to system")
            return
        
        # Restore original methods
        for method_name, method in self.original_methods.items():
            setattr(self.system, method_name, method)
        
        # Remove added methods
        if hasattr(self.system, 'get_recursion_tracker'):
            delattr(self.system, 'get_recursion_tracker')
            
        if hasattr(self.system, 'get_transition_analyzer'):
            delattr(self.system, 'get_transition_analyzer')
            
        if hasattr(self.system, 'get_fractal_analyzer'):
            delattr(self.system, 'get_fractal_analyzer')
            
        if hasattr(self.system, 'get_meta_pattern_detector'):
            delattr(self.system, 'get_meta_pattern_detector')
            
        if hasattr(self.system, 'generate_recursion_report'):
            delattr(self.system, 'generate_recursion_report')
        
        self.attached = False
        logger.info("Recursion components detached from UTCHSSystem")
    
    def _attach_advance_tick(self):
        """Attach enhanced advance_tick method."""
        original_advance_tick = self.system.advance_tick
        tracking_interval = self.tracking_interval
        
        def enhanced_advance_tick():
            """Enhanced advance_tick with recursion tracking."""
            result = original_advance_tick()
            
            # Only track at specified intervals to reduce overhead
            if self.system.current_tick % tracking_interval == 0:
                self._track_current_position(self.system.current_tick)
            
            # Run analysis at specified intervals
            if self.system.current_tick % self.analysis_interval == 0:
                self._run_recursion_analysis(self.system.current_tick)
            
            # Run visualization at specified intervals
            if self.system.current_tick % self.visualization_interval == 0:
                self._run_recursion_visualization(self.system.current_tick)
            
            return result
        
        self.system.advance_tick = enhanced_advance_tick
    
    def _attach_record_state(self):
        """Attach enhanced _record_state method."""
        original_record_state = self.system._record_state
        
        def enhanced_record_state():
            """Enhanced _record_state with recursion metrics."""
            state = original_record_state()
            
            # Add recursion metrics if available
            if state and self.system.current_tick % self.tracking_interval == 0:
                recursion_metrics = self._get_recursion_metrics()
                if recursion_metrics:
                    state['recursion_metrics'] = recursion_metrics
            
            return state
        
        self.system._record_state = enhanced_record_state
    
    def _attach_update_system_metrics(self):
        """Attach enhanced _update_system_metrics method."""
        original_update_system_metrics = self.system._update_system_metrics
        
        def enhanced_update_system_metrics():
            """Enhanced _update_system_metrics with recursion metrics."""
            original_update_system_metrics()
            
            # Update recursion-related metrics
            if self.system.current_tick % self.tracking_interval == 0:
                self._update_recursion_metrics()
        
        self.system._update_system_metrics = enhanced_update_system_metrics
    
    def _attach_analyze_system_state(self):
        """Attach enhanced analyze_system_state method."""
        original_analyze_system_state = self.system.analyze_system_state
        
        def enhanced_analyze_system_state():
            """Enhanced analyze_system_state with recursion analysis."""
            analysis = original_analyze_system_state()
            
            # Add recursion analysis
            recursion_analysis = self._analyze_recursion()
            if recursion_analysis:
                analysis['recursion_analysis'] = recursion_analysis
            
            return analysis
        
        self.system.analyze_system_state = enhanced_analyze_system_state
    
    def _track_current_position(self, tick):
        """
        Track the current position in the recursion tracker.
        
        Args:
            tick: Current system tick
        """
        try:
            # Get current position from the system
            current_torus = self.system.tori[self.system.current_torus_idx]
            current_structure = current_torus.get_current_structure()
            current_cycle = current_structure.get_current_cycle()
            current_position = current_cycle.get_current_position()
            
            # Track position in the recursion tracker
            recursion_depth = self.system.phase_recursion_depth
            self.recursion_tracker.track_position(current_position, recursion_depth, tick)
            
            logger.debug(f"Tracked position {current_position.number} at recursion depth {recursion_depth}")
            
        except Exception as e:
            logger.error(f"Error tracking position: {str(e)}")
    
    def _run_recursion_analysis(self, tick):
        """
        Run recursion analysis at specified intervals.
        
        Args:
            tick: Current system tick
        """
        try:
            logger.info(f"Running recursion analysis at tick {tick}")
            
            # Analyze P13 seventh cycle transformation
            p13_analysis = self.transition_analyzer.analyze_p13_seventh_cycle_transformation()
            
            # Analyze octave transitions
            octave_analysis = self.transition_analyzer.analyze_octave_transitions()
            
            # Detect phi resonances
            phi_resonance = self.transition_analyzer.detect_phi_resonances()
            
            # Analyze fractal properties
            fractal_dimension = self.fractal_analyzer.calculate_fractal_dimension()
            multi_scale_entropy = self.fractal_analyzer.calculate_multi_scale_entropy()
            self_similarity = self.fractal_analyzer.calculate_self_similarity()
            
            # Analyze meta-patterns
            meta_pattern_analysis = self.meta_pattern_detector.detect_meta_patterns(
                self.recursion_tracker.position_history,
                system_state=self._get_system_state()
            )
            
            # Log significant findings
            if p13_analysis and p13_analysis.get('detected', False):
                logger.info(f"P13 Seventh Cycle Transformation detected at tick {tick}")
            
            if meta_pattern_analysis and meta_pattern_analysis.get('detected', False):
                logger.info(f"Meta-pattern detected at tick {tick} with strength {meta_pattern_analysis.get('meta_cycle_strength', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Error running recursion analysis: {str(e)}")
    
    def _run_recursion_visualization(self, tick):
        """
        Run recursion visualization at specified intervals.
        
        Args:
            tick: Current system tick
        """
        try:
            # Create visualization directory
            output_dir = self.config.get('output_dir', 'recursion_visualization')
            os.makedirs(output_dir, exist_ok=True)
            tick_dir = os.path.join(output_dir, f"tick_{tick}")
            os.makedirs(tick_dir, exist_ok=True)
            
            # Create visualization instance if needed
            if not hasattr(self, 'visualizer'):
                self.visualizer = RecursionVisualizer(self.recursion_tracker, self.config)
            
            # Generate visualizations
            self.visualizer.visualize_position_across_depths(
                10, tick, save=True, 
                filename=os.path.join(tick_dir, f"position_10_depths_tick_{tick}.png")
            )
            
            self.visualizer.visualize_position_across_depths(
                13, tick, save=True, 
                filename=os.path.join(tick_dir, f"position_13_depths_tick_{tick}.png")
            )
            
            logger.info(f"Generated recursion visualizations at tick {tick} in {tick_dir}")
            
        except Exception as e:
            logger.error(f"Error generating recursion visualizations: {str(e)}")
    
    def _get_recursion_metrics(self):
        """
        Get current recursion metrics.
        
        Returns:
            Dictionary with recursion metrics
        """
        try:
            # Get current recursion transitions
            transitions = self.recursion_tracker.get_recursion_transitions()
            
            # Count phi resonances
            phi_resonance_count = sum(
                1 for t in transitions 
                if t.get('phi_phase_resonance') or t.get('phi_energy_resonance')
            )
            
            # Get fractal metrics
            self_similarity = self.fractal_analyzer.calculate_self_similarity()
            fractal_dimension = self.fractal_analyzer.calculate_fractal_dimension()
            
            # Get meta-pattern metrics
            meta_pattern = self.meta_pattern_detector.detect_meta_patterns(
                self.recursion_tracker.position_history
            )
            
            return {
                'transitions_count': len(transitions),
                'phi_resonance_count': phi_resonance_count,
                'phi_resonance_percentage': phi_resonance_count / max(1, len(transitions)),
                'self_similarity': self_similarity.get('self_similarity') if self_similarity else None,
                'fractal_dimension': fractal_dimension.get('dimension') if fractal_dimension else None,
                'meta_pattern_detected': meta_pattern.get('detected', False),
                'meta_pattern_strength': meta_pattern.get('meta_cycle_strength', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting recursion metrics: {str(e)}")
            return {}
    
    def _update_recursion_metrics(self):
        """Update system metrics with recursion-related data."""
        try:
            # Get recursion metrics
            metrics = self._get_recursion_metrics()
            
            # Add fractal dimension as system metric
            if 'fractal_dimension' in metrics and metrics['fractal_dimension'] is not None:
                self.system.fractal_dimension = metrics['fractal_dimension']
            
            # Add self-similarity as system metric
            if 'self_similarity' in metrics and metrics['self_similarity'] is not None:
                self.system.self_similarity = metrics['self_similarity']
            
            # Add meta-pattern detection as system metric
            if 'meta_pattern_detected' in metrics:
                self.system.meta_pattern_detected = metrics['meta_pattern_detected']
                self.system.meta_pattern_strength = metrics.get('meta_pattern_strength', 0)
            
        except Exception as e:
            logger.error(f"Error updating recursion metrics: {str(e)}")
    
    def _analyze_recursion(self):
        """
        Analyze recursion patterns in the system.
        
        Returns:
            Dictionary with recursion analysis
        """
        try:
            # Analyze transitions
            transitions = self.recursion_tracker.get_recursion_transitions()
            transition_summary = self._summarize_transitions(transitions)
            
            # Analyze phi resonances
            phi_resonance = self.transition_analyzer.detect_phi_resonances()
            
            # Analyze fractal properties
            fractal_dimension = self.fractal_analyzer.calculate_fractal_dimension()
            self_similarity = self.fractal_analyzer.calculate_self_similarity()
            
            # Analyze meta-patterns
            meta_pattern = self.meta_pattern_detector.detect_meta_patterns(
                self.recursion_tracker.position_history
            )
            
            return {
                'transitions': transition_summary,
                'phi_resonance': phi_resonance,
                'fractal_dimension': fractal_dimension,
                'self_similarity': self_similarity,
                'meta_pattern': meta_pattern
            }
            
        except Exception as e:
            logger.error(f"Error analyzing recursion: {str(e)}")
            return {}
    
    def _summarize_transitions(self, transitions):
        """
        Summarize recursion transitions.
        
        Args:
            transitions: List of transition dictionaries
            
        Returns:
            Dictionary with transition summary
        """
        if not transitions:
            return {'count': 0, 'message': 'No transitions detected'}
        
        # Count transitions by recursion depth
        depth_counts = {}
        for t in transitions:
            depth = t.get('recursion_depth', 0)
            if depth not in depth_counts:
                depth_counts[depth] = 0
            depth_counts[depth] += 1
        
        # Count phi resonances
        phi_count = sum(
            1 for t in transitions 
            if t.get('phi_phase_resonance') or t.get('phi_energy_resonance')
        )
        
        return {
            'count': len(transitions),
            'phi_resonance_count': phi_count,
            'phi_resonance_percentage': phi_count / len(transitions),
            'by_recursion_depth': depth_counts,
            'most_recent': transitions[-1] if transitions else None
        }
    
    def _get_system_state(self):
        """
        Get current system state for adaptive corrections.
        
        Returns:
            Dictionary with system state
        """
        return {
            'tick': self.system.current_tick,
            'recursion_depth': self.system.phase_recursion_depth,
            'global_coherence': self.system.global_coherence,
            'global_stability': self.system.global_stability,
            'energy_level': self.system.energy_level
        }
    
    def _get_recursion_tracker(self):
        """
        Get recursion tracker instance.
        
        Returns:
            RecursionTracker singleton instance
        """
        return self.recursion_tracker
    
    def _get_transition_analyzer(self):
        """
        Get transition analyzer instance.
        
        Returns:
            TransitionAnalyzer instance
        """
        return self.transition_analyzer
    
    def _get_fractal_analyzer(self):
        """
        Get fractal analyzer instance.
        
        Returns:
            FractalAnalyzer instance
        """
        return self.fractal_analyzer
    
    def _get_meta_pattern_detector(self):
        """
        Get meta-pattern detector instance.
        
        Returns:
            MetaPatternDetector instance
        """
        return self.meta_pattern_detector
    
    def _generate_recursion_report(self, output_file=None):
        """
        Generate comprehensive recursion analysis report.
        
        Args:
            output_file: Output file path (optional)
            
        Returns:
            Report as a string
        """
        if output_file is None:
            output_dir = self.config.get('output_dir', 'recursion_analysis')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"recursion_report_tick_{self.system.current_tick}.txt")
        
        # Use RecursionIntegrator's report generation capability
        return self.recursion_integrator.generate_report(output_file)


def integrate_recursion_tracking(system, config=None):
    """
    Integrate recursion tracking with a UTCHSSystem.
    
    This is the primary entry point for enabling recursion tracking
    in a UTCHSSystem instance.
    
    Args:
        system: UTCHSSystem instance
        config: Configuration dictionary (optional)
        
    Returns:
        UTCHSSystemIntegrator instance
    """
    integrator = UTCHSSystemIntegrator(system, config)
    integrator.attach_components()
    
    logger.info("Recursion tracking integrated with UTCHSSystem")
    return integrator


def create_default_configuration():
    """
    Create default configuration for recursion integration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'tracking_interval': 1,           # Track every tick by default
        'analysis_interval': 100,         # Analyze every 100 ticks
        'visualization_interval': 500,    # Visualize every 500 ticks
        'output_dir': 'recursion_output', # Output directory
        
        # RecursionTracker configuration
        'recursion': {
            'max_history_length': 1000,
            'max_recursion_depth': 7
        },
        
        # Meta-pattern detector configuration
        'meta_pattern': {
            'correlation_threshold': 0.7,
            'phase_coherence_threshold': 0.6,
            'energy_pattern_threshold': 0.65,
            'max_recursion_order': 5,
            'enable_phase_locking': True,
            'enable_coherence_gating': True
        },
        
        # Phase-locking configuration
        'phase_lock': {
            'lock_strength': 0.7,
            'phase_tolerance': 0.1
        },
        
        # Coherence gate configuration
        'coherence_gate': {
            'base_threshold': 0.7,
            'recursion_factor': 0.1,
            'min_threshold': 0.4,
            'enable_adaptive_threshold': True
        }
    } 