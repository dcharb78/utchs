"""
Recursion integration module for the UTCHS framework.

This module implements the RecursionIntegrator class, which integrates the
recursion tracking components with the existing UTCHSSystem.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import matplotlib.pyplot as plt

from ..utils.logging_config import get_logger
from ..core.recursion_tracker import RecursionTracker
from ..core.transition_analyzer import TransitionAnalyzer
from ..core.fractal_analyzer import FractalAnalyzer
from ..visualization.recursion_vis import RecursionVisualizer
from .meta_pattern_detector import MetaPatternDetector

logger = get_logger(__name__)

class RecursionIntegrator:
    """
    Integrates recursion tracking, transition analysis, and fractal analysis.
    
    This class provides a unified interface for recursion tracking, coordinating
    between the RecursionTracker, TransitionAnalyzer, and FractalAnalyzer.
    """
    
    def __init__(self, system, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the recursion integrator.
        
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
        
        # Last analyzed tick
        self.last_analyzed_tick = 0
        
        # Last visualized tick
        self.last_visualized_tick = 0
        
        logger.info("RecursionIntegrator initialized")
    
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
    
    def _run_analysis(self, tick: int) -> None:
        """
        Run analysis on recursion data.
        
        Args:
            tick: Current system tick
        """
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
        scaling_invariance = self.fractal_analyzer.calculate_scaling_invariance()
        
        # Analyze meta-patterns
        meta_pattern_analysis = self.meta_pattern_detector.detect_meta_patterns(
            self.recursion_tracker.position_history
        )
        
        cycle6_meta_analysis = None
        if meta_pattern_analysis['detected']:
            cycle6_meta_analysis = self.meta_pattern_detector.analyze_cycle6_meta_transition(
                self.recursion_tracker.position_history
            )
        
        # Compile analysis results
        analysis_results = {
            'p13_seventh_cycle': p13_analysis,
            'octave_transitions': octave_analysis,
            'phi_resonance': phi_resonance,
            'fractal_dimension': fractal_dimension,
            'self_similarity': self_similarity,
            'multi_scale_entropy': multi_scale_entropy,
            'meta_pattern': meta_pattern_analysis,
            'cycle6_meta_transition': cycle6_meta_analysis
        }
        
        # Log important findings
        if p13_analysis.get('detected', False):
            logger.info(f"P13 Seventh Cycle Transformation detected at tick {tick}")
            
            # Check for golden ratio resonance
            if p13_analysis.get('phi_phase_resonance', False) or p13_analysis.get('phi_energy_resonance', False):
                logger.info(f"Golden ratio resonance detected in P13 transformation")
        
        # Store analysis timestamp
        self.last_analyzed_tick = tick
    
    def _run_visualization(self, tick: int) -> None:
        """
        Run visualization of recursion data.
        
        Args:
            tick: Current system tick
        """
        logger.info(f"Generating recursion visualizations at tick {tick}")
        
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
        
        # Visualize P13 seventh cycle transformation
        p13_transform_vis = self.visualizer.visualize_p13_seventh_cycle(
            self.transition_analyzer, save=True,
            filename=os.path.join(tick_dir, f"p13_transformation_tick_{tick}.png")
        )
        plt.close(p13_transform_vis)
        
        # Visualize fractal metrics
        fractal_vis = self.visualizer.visualize_fractal_metrics(
            self.fractal_analyzer, save=True,
            filename=os.path.join(tick_dir, f"fractal_metrics_tick_{tick}.png")
        )
        plt.close(fractal_vis)
        
        # Store visualization timestamp
        self.last_visualized_tick = tick
    
    def generate_report(self, output_file: str = "recursion_analysis_report.txt") -> None:
        """
        Generate a comprehensive report of recursion analysis.
        
        Args:
            output_file: Output file path
        """
        logger.info(f"Generating recursion analysis report to {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("=======================================================\n")
            f.write("UTCHS RECURSIVE TRANSITION ANALYSIS REPORT\n")
            f.write("=======================================================\n\n")
            
            # System information
            f.write(f"System Tick: {self.system.current_tick}\n")
            f.write(f"Recursion Depth: {self.system.phase_recursion_depth}\n")
            f.write(f"Number of Tori: {len(self.system.tori)}\n\n")
            
            # P13 Seventh Cycle Transformation
            f.write("-------------------------------------------------------\n")
            f.write("P13 SEVENTH CYCLE TRANSFORMATION ANALYSIS\n")
            f.write("-------------------------------------------------------\n")
            
            p13_analysis = self.transition_analyzer.detected_transitions['p13_seventh_cycle']
            
            if not p13_analysis:
                f.write("No P13 transformations detected yet.\n\n")
            else:
                f.write(f"Detected {len(p13_analysis)} P13 transformations.\n\n")
                
                # Most recent transformation
                recent = p13_analysis[-1]
                f.write("Most recent transformation:\n")
                f.write(f"  Tick: {recent['from_tick']} -> {recent['to_tick']}\n")
                f.write(f"  Phase Shift: {recent['phase_shift']:.4f}\n")
                f.write(f"  Energy Ratio: {recent['energy_ratio']:.4f}\n")
                f.write(f"  Phi Resonance: {'Yes' if recent['phi_phase_resonance'] or recent['phi_energy_resonance'] else 'No'}\n\n")
            
            # Octave Transitions
            f.write("-------------------------------------------------------\n")
            f.write("OCTAVE TRANSITIONS ANALYSIS\n")
            f.write("-------------------------------------------------------\n")
            
            octave_transitions = self.transition_analyzer.detected_transitions['octave_transitions']
            
            if not octave_transitions:
                f.write("No octave transitions analyzed yet.\n\n")
            else:
                for depth_summary in octave_transitions:
                    f.write(f"Recursion Depth {depth_summary['recursion_depth']}:\n")
                    f.write(f"  Transitions: {depth_summary['transition_count']}\n")
                    f.write(f"  Avg Phase Shift: {depth_summary['avg_phase_shift']:.4f}\n")
                    f.write(f"  Avg Energy Ratio: {depth_summary['avg_energy_ratio']:.4f}\n")
                    f.write(f"  Phi Resonance %: {depth_summary['phi_resonance_percentage']*100:.1f}%\n\n")
            
            # Phi Resonances
            f.write("-------------------------------------------------------\n")
            f.write("PHI RESONANCE ANALYSIS\n")
            f.write("-------------------------------------------------------\n")
            
            phi_resonances = self.transition_analyzer.detected_transitions['phi_resonances']
            
            if not phi_resonances:
                f.write("No phi resonance analysis available yet.\n\n")
            else:
                recent_resonance = phi_resonances[-1]
                f.write(f"Phi resonance detected: {'Yes' if recent_resonance['detected'] else 'No'}\n")
                f.write(f"Energy strength: {recent_resonance['phi_energy_strength']:.2f}\n")
                f.write(f"Phase strength: {recent_resonance['invphi_phase_strength']:.2f}\n\n")
            
            # Fractal Analysis
            f.write("-------------------------------------------------------\n")
            f.write("FRACTAL ANALYSIS\n")
            f.write("-------------------------------------------------------\n")
            
            if 'fractal_dimension' in self.fractal_analyzer.fractal_metrics:
                fd = self.fractal_analyzer.fractal_metrics['fractal_dimension']
                f.write(f"Fractal Dimension: {fd['dimension']:.4f} (R²={fd['r_squared']:.4f})\n\n")
            else:
                f.write("No fractal dimension data available yet.\n\n")
            
            if 'self_similarity' in self.fractal_analyzer.fractal_metrics:
                ss = self.fractal_analyzer.fractal_metrics['self_similarity']
                f.write(f"Self-Similarity: {ss['self_similarity']:.4f}\n")
                f.write(f"Phi Scaling %: {ss['phi_scaling_percentage']*100:.1f}%\n\n")
            else:
                f.write("No self-similarity data available yet.\n\n")
            
            if 'scaling_invariance' in self.fractal_analyzer.fractal_metrics:
                si = self.fractal_analyzer.fractal_metrics['scaling_invariance']
                f.write(f"Avg Scaling Factor: {si['average_scaling_factor']:.4f}\n")
                f.write(f"Phi Deviation: {si['phi_deviation']:.4f}\n")
                f.write(f"Phi Resonance: {'Yes' if si['phi_resonance'] else 'No'}\n")
                f.write(f"Invariant Positions: {', '.join(map(str, si['invariant_positions']))}\n\n")
            else:
                f.write("No scaling invariance data available yet.\n\n")
            
            # Meta-Pattern Analysis
            f.write("-------------------------------------------------------\n")
            f.write("META-PATTERN ANALYSIS\n")
            f.write("-------------------------------------------------------\n")
            
            meta_pattern = self.meta_pattern_detector.detect_meta_patterns(
                self.recursion_tracker.position_history
            )
            
            f.write(f"Meta-pattern detected: {'Yes' if meta_pattern.get('detected', False) else 'No'}\n")
            f.write(f"Meta-cycle strength: {meta_pattern.get('meta_cycle_strength', 0):.4f}\n")
            
            if meta_pattern.get('detected', False):
                f.write(f"Position 3 -> Cycle 6 correlation: {meta_pattern.get('position3_cycle6_correlation', 0):.4f}\n")
                f.write(f"Position 6 -> Cycle 9 correlation: {meta_pattern.get('position6_cycle9_correlation', 0):.4f}\n")
                f.write(f"Position 9 -> Cycle 12 correlation: {meta_pattern.get('position9_cycle12_correlation', 0):.4f}\n")
                
                # Include cycle 6 meta-transition details if available
                if self.meta_pattern_detector.analyze_cycle6_meta_transition(
                    self.recursion_tracker.position_history
                ):
                    c6_meta = self.meta_pattern_detector.analyze_cycle6_meta_transition(
                        self.recursion_tracker.position_history
                    )
                    f.write("\nCycle 6 Meta-Transition Details:\n")
                    f.write(f"  Confidence: {c6_meta.get('confidence', 0):.4f}\n")
                    f.write(f"  Position 3 Resonance: {c6_meta.get('position3_resonance', {}).get('strength', 0):.4f}\n")
                    
                    # Include energy pattern if detected
                    energy_pattern = c6_meta.get('energy_pattern', {})
                    if energy_pattern.get('pattern_detected', False):
                        f.write(f"  Dominant Energy Pattern: {energy_pattern.get('dominant_pattern', 'unknown')}\n")
                        f.write(f"  Pattern Strength: {energy_pattern.get('dominant_score', 0):.4f}\n")
                    
                    # Include phi resonance if detected
                    phi_res = c6_meta.get('phi_resonance', {})
                    if phi_res.get('detected', False):
                        f.write(f"  Phi Resonance: Yes (strength: {phi_res.get('strength', 0):.4f})\n")
            
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
            
            # Determine key insights
            insights = []
            
            # Check for P13 transformations with phi resonance
            phi_resonant_p13 = [
                t for t in p13_analysis 
                if t['phi_phase_resonance'] or t['phi_energy_resonance']
            ]
            
            if phi_resonant_p13:
                insights.append(f"Detected {len(phi_resonant_p13)} P13 transformations with golden ratio resonance")
            
            # Check for scaling invariance
            if 'scaling_invariance' in self.fractal_analyzer.fractal_metrics:
                si = self.fractal_analyzer.fractal_metrics['scaling_invariance']
                if si['phi_resonance']:
                    insights.append(f"System exhibits phi-scaling with deviation of only {si['phi_deviation']:.4f}")
                    insights.append(f"Positions {', '.join(map(str, si['invariant_positions']))} show invariant scaling properties")
            
            # Output insights
            if insights:
                f.write("Key insights:\n")
                for i, insight in enumerate(insights, 1):
                    f.write(f"{i}. {insight}\n")
            else:
                f.write("Insufficient data for conclusive insights. Continue running the system to gather more data.\n")
        
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

    def analyze_system(self, system):
        """
        Perform comprehensive analysis of the system's recursive structure.
        
        Args:
            system: UTCHS system object
            
        Returns:
            Dictionary with analysis results
        """
        # Analyze transitions
        p13_analysis = self.transition_analyzer.analyze_p13_seventh_cycle_transformation()
        octave_analysis = self.transition_analyzer.analyze_octave_transitions()
        phi_resonance = self.transition_analyzer.detect_phi_resonances()
        
        # Calculate fractal metrics
        fractal_dim = self.fractal_analyzer.calculate_fractal_dimension()
        self_similarity = self.fractal_analyzer.calculate_self_similarity()
        multi_scale_entropy = self.fractal_analyzer.calculate_multi_scale_entropy()
        
        # Analyze meta-patterns
        meta_pattern_analysis = self.meta_pattern_detector.detect_meta_patterns(
            self.recursion_tracker.position_history
        )
        
        cycle6_meta_analysis = None
        if meta_pattern_analysis['detected']:
            cycle6_meta_analysis = self.meta_pattern_detector.analyze_cycle6_meta_transition(
                self.recursion_tracker.position_history
            )
        
        # Compile analysis results
        analysis_results = {
            'p13_seventh_cycle': p13_analysis,
            'octave_transitions': octave_analysis,
            'phi_resonance': phi_resonance,
            'fractal_dimension': fractal_dim,
            'self_similarity': self_similarity,
            'multi_scale_entropy': multi_scale_entropy,
            'meta_pattern': meta_pattern_analysis,
            'cycle6_meta_transition': cycle6_meta_analysis
        }
        
        return analysis_results

    def generate_report(self, system, file_path=None):
        """
        Generate a comprehensive report of recursion analysis.
        
        Args:
            system: UTCHS system object
            file_path: Output file path (optional)
            
        Returns:
            Report as a string
        """
        if file_path is None:
            file_path = os.path.join(os.getcwd(), "recursion_final_report.txt")
        
        analysis = self.analyze_system(system)
        
        # Create report header
        report = "=======================================================\n"
        report += "UTCHS RECURSIVE TRANSITION ANALYSIS REPORT\n"
        report += "=======================================================\n\n"
        
        # System info
        report += f"System Tick: {system.tick}\n"
        report += f"Recursion Depth: {system.max_recursion_depth}\n"
        report += f"Number of Tori: {len(system.tori)}\n\n"
        
        # P13 Seventh Cycle Transformation
        report += "-------------------------------------------------------\n"
        report += "P13 SEVENTH CYCLE TRANSFORMATION ANALYSIS\n"
        report += "-------------------------------------------------------\n"
        
        p13_analysis = analysis['p13_seventh_cycle']
        if isinstance(p13_analysis, list):
            report += f"Detected {len(p13_analysis)} P13 transformations.\n\n"
            if p13_analysis:
                report += "Most recent transformation:\n"
                latest = p13_analysis[-1]
                report += f"  Tick: {latest['from_tick']} -> {latest['to_tick']}\n"
                report += f"  Phase Shift: {latest.get('phase_shift', 0):.4f}\n"
                report += f"  Energy Ratio: {latest.get('energy_ratio', 0):.4f}\n"
                report += f"  Phi Resonance: {'Yes' if latest.get('phi_phase_resonance') or latest.get('phi_energy_resonance') else 'No'}\n"
        else:
            report += f"P13 transformation detection: {p13_analysis.get('message', 'No data')}\n"
        
        report += "\n"
        
        # Octave Transitions
        report += "-------------------------------------------------------\n"
        report += "OCTAVE TRANSITIONS ANALYSIS\n"
        report += "-------------------------------------------------------\n"
        
        octave_transitions = analysis['octave_transitions']
        if octave_transitions:
            for depth_data in octave_transitions:
                report += f"Recursion Depth {depth_data['recursion_depth']}:\n"
                report += f"  Transitions: {depth_data['transition_count']}\n"
                report += f"  Avg Phase Shift: {depth_data.get('avg_phase_shift', 0):.4f}\n"
                report += f"  Avg Energy Ratio: {depth_data.get('avg_energy_ratio', 0):.4f}\n"
                phi_pct = depth_data.get('phi_resonance_percentage', 0) * 100
                report += f"  Phi Resonance %: {phi_pct:.1f}%\n\n"
        else:
            report += "No octave transitions detected.\n\n"
        
        # Phi Resonance
        report += "-------------------------------------------------------\n"
        report += "PHI RESONANCE ANALYSIS\n"
        report += "-------------------------------------------------------\n"
        
        phi_res = analysis['phi_resonance']
        report += f"Phi resonance detected: {'Yes' if phi_res.get('detected', False) else 'No'}\n"
        report += f"Energy strength: {phi_res.get('phi_energy_strength', 0):.2f}\n"
        report += f"Phase strength: {phi_res.get('invphi_phase_strength', 0):.2f}\n\n"
        
        # Fractal Analysis
        report += "-------------------------------------------------------\n"
        report += "FRACTAL ANALYSIS\n"
        report += "-------------------------------------------------------\n"
        
        fractal_dim = analysis['fractal_dimension']
        report += f"Fractal Dimension: {fractal_dim.get('dimension', 0):.4f} (R²={fractal_dim.get('r_squared', 0):.4f})\n\n"
        
        self_sim = analysis['self_similarity']
        report += f"Self-Similarity: {self_sim.get('self_similarity', 'nan')}\n"
        phi_scaling = self_sim.get('phi_scaling_percentage', 0) * 100
        report += f"Phi Scaling %: {phi_scaling:.1f}%\n\n"
        
        avg_scaling = self_sim.get('average_scaling', 0)
        phi_dev = abs(avg_scaling - (1 + 5**0.5)/2) if avg_scaling > 0 else float('inf')
        report += f"Avg Scaling Factor: {avg_scaling:.4f}\n"
        report += f"Phi Deviation: {phi_dev if not np.isinf(phi_dev) else 'inf'}\n"
        report += f"Phi Resonance: {'Yes' if phi_dev < 0.2 else 'No'}\n"
        report += f"Invariant Positions: {', '.join(map(str, self_sim.get('invariant_positions', [])))}\n\n"
        
        # Meta-Pattern Analysis
        report += "-------------------------------------------------------\n"
        report += "META-PATTERN ANALYSIS\n"
        report += "-------------------------------------------------------\n"
        
        meta_pattern = analysis['meta_pattern']
        report += f"Meta-pattern detected: {'Yes' if meta_pattern.get('detected', False) else 'No'}\n"
        report += f"Meta-cycle strength: {meta_pattern.get('meta_cycle_strength', 0):.4f}\n"
        
        if meta_pattern.get('detected', False):
            report += f"Position 3 -> Cycle 6 correlation: {meta_pattern.get('position3_cycle6_correlation', 0):.4f}\n"
            report += f"Position 6 -> Cycle 9 correlation: {meta_pattern.get('position6_cycle9_correlation', 0):.4f}\n"
            report += f"Position 9 -> Cycle 12 correlation: {meta_pattern.get('position9_cycle12_correlation', 0):.4f}\n"
            
            # Include cycle 6 meta-transition details if available
            if analysis['cycle6_meta_transition']:
                c6_meta = analysis['cycle6_meta_transition']
                report += "\nCycle 6 Meta-Transition Details:\n"
                report += f"  Confidence: {c6_meta.get('confidence', 0):.4f}\n"
                report += f"  Position 3 Resonance: {c6_meta.get('position3_resonance', {}).get('strength', 0):.4f}\n"
                
                # Include energy pattern if detected
                energy_pattern = c6_meta.get('energy_pattern', {})
                if energy_pattern.get('pattern_detected', False):
                    report += f"  Dominant Energy Pattern: {energy_pattern.get('dominant_pattern', 'unknown')}\n"
                    report += f"  Pattern Strength: {energy_pattern.get('dominant_score', 0):.4f}\n"
                
                # Include phi resonance if detected
                phi_res = c6_meta.get('phi_resonance', {})
                if phi_res.get('detected', False):
                    report += f"  Phi Resonance: Yes (strength: {phi_res.get('strength', 0):.4f})\n"
        
        report += "\n"
        
        # Visualization Paths
        report += "-------------------------------------------------------\n"
        report += "VISUALIZATION PATHS\n"
        report += "-------------------------------------------------------\n"
        
        report += f"Output Directory: {self.output_dir}\n"
        report += f"Last Visualization: Tick {self.last_visualized_tick}\n\n"
        
        # Conclusion
        report += "=======================================================\n"
        report += "CONCLUSION\n"
        report += "=======================================================\n\n"
        
        if system.tick < 500:
            report += "Insufficient data for conclusive insights. Continue running the system to gather more data.\n\n"
        elif meta_pattern.get('detected', False):
            report += "Meta-pattern detected at cycle 6, confirming the theoretical prediction of recursive 3-6-9 patterns.\n"
            report += "The system has established a second-order 3-6-9 pattern where cycle 6 becomes a meta-position 3,\n"
            report += "cycle 9 becomes a meta-position 6, and cycle 12 will likely become a meta-position 9.\n\n"
            report += "This recursive structure demonstrates how the system generates complexity\n"
            report += "while maintaining the fundamental 3-6-9 organizing principle across multiple scales.\n\n"
        else:
            report += "Analysis shows typical recursion patterns but no clear meta-pattern emergence yet.\n"
            report += "Continue monitoring for emergence of the recursive 3-6-9 pattern at cycle 6.\n\n"
        
        # Write report to file
        with open(file_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Generated recursion analysis report: {file_path}")
        
        return report

    def create_visualizations(self, system, tick_interval=100):
        """
        Create recursion analysis visualizations.
        
        Args:
            system: UTCHS system object
            tick_interval: Tick interval for creating visualizations
            
        Returns:
            Dictionary with visualization paths
        """
        if system.tick % tick_interval != 0:
            return None
        
        # Create visualization directory for this tick
        tick_dir = os.path.join(self.output_dir, f"tick_{system.tick}")
        os.makedirs(tick_dir, exist_ok=True)
        
        # Create visualizations
        vis_paths = {}
        
        # P13 transformation visualization
        p13_path = os.path.join(tick_dir, f"p13_transformation_tick_{system.tick}.png")
        self.visualizer.plot_p13_transformation(self.transition_analyzer, p13_path)
        vis_paths['p13_transformation'] = p13_path
        
        # Position depths visualization
        pos13_path = os.path.join(tick_dir, f"position_13_depths_tick_{system.tick}.png")
        self.visualizer.plot_position_depths(self.recursion_tracker, 13, pos13_path)
        vis_paths['position_13_depths'] = pos13_path
        
        pos10_path = os.path.join(tick_dir, f"position_10_depths_tick_{system.tick}.png")
        self.visualizer.plot_position_depths(self.recursion_tracker, 10, pos10_path)
        vis_paths['position_10_depths'] = pos10_path
        
        # Fractal metrics visualization
        fractal_path = os.path.join(tick_dir, f"fractal_metrics_tick_{system.tick}.png")
        self.visualizer.plot_fractal_metrics(self.fractal_analyzer, fractal_path)
        vis_paths['fractal_metrics'] = fractal_path
        
        # Phi scaling visualization
        phi_path = os.path.join(tick_dir, f"phi_scaling_tick_{system.tick}.png")
        self.visualizer.plot_phi_scaling(self.transition_analyzer, phi_path)
        vis_paths['phi_scaling'] = phi_path
        
        # Meta-pattern visualization if detected
        meta_pattern_analysis = self.meta_pattern_detector.detect_meta_patterns(
            self.recursion_tracker.position_history
        )
        
        if meta_pattern_analysis['detected']:
            meta_path = os.path.join(tick_dir, f"meta_pattern_tick_{system.tick}.png")
            self._create_meta_pattern_visualization(meta_path)
            vis_paths['meta_pattern'] = meta_path
        
        self.last_visualized_tick = system.tick
        logger.info(f"Created visualizations for tick {system.tick} in {tick_dir}")
        
        return vis_paths
    
    def _create_meta_pattern_visualization(self, output_path):
        """
        Create visualization for meta-pattern analysis.
        
        This is a placeholder that will be implemented in the
        meta_pattern_vis.py module when it's developed.
        
        Args:
            output_path: Output file path
        """
        # This will be replaced with proper implementation
        # when the meta_pattern_vis.py module is created
        plt.figure(figsize=(10, 8))
        plt.title("Meta-Pattern Analysis (Placeholder)")
        plt.text(0.5, 0.5, "Meta-pattern visualization\nwill be implemented in Phase 2", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close() 