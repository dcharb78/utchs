"""
Transition analyzer module for the UTCHS framework.

This module implements the TransitionAnalyzer class, which detects and analyzes
patterns between scales and recursive levels in the UTCHS framework.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict

from ..utils.logging_config import get_logger
from .recursion_tracker import RecursionTracker

logger = get_logger(__name__)

class TransitionAnalyzer:
    """
    Analyzes transitions between scales and recursive levels.
    
    This class focuses on detecting patterns between scales, especially
    the 7th cycle P13 transformation, and understanding how transformations
    propagate through the recursive structure.
    """
    
    def __init__(self, recursion_tracker: RecursionTracker, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the transition analyzer.
        
        Args:
            recursion_tracker: RecursionTracker instance containing position history
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.recursion_tracker = recursion_tracker
        
        # Golden ratio (φ) for resonance detection
        self.phi = (1 + np.sqrt(5)) / 2
        
        # Keep track of detected transitions
        self.detected_transitions = {
            'p13_seventh_cycle': [],
            'octave_transitions': [],
            'phi_resonances': []
        }
        
        logger.info("TransitionAnalyzer initialized")
    
    def analyze_p13_seventh_cycle_transformation(self) -> Dict:
        """
        Analyze the P13 transformation at the 7th cycle.
        
        Returns:
            Dictionary with transformation metrics
        """
        # Get P13 history across all recursion levels
        p13_history = self.recursion_tracker.get_position_history(13)
        
        # Filter for cycle 7 only
        cycle7_p13 = [
            data for data in p13_history 
            if data['absolute_position']['cycle'] == 7
        ]
        
        # We need a minimum number of P13 observations in cycle 7
        if len(cycle7_p13) < 2:
            return {'detected': False, 'message': 'Not enough P13 data in 7th cycle'}
        
        # Get previous cycle (cycle 6) P13 data for comparison
        cycle6_p13 = [
            data for data in p13_history 
            if data['absolute_position']['cycle'] == 6
        ]
        
        if not cycle6_p13:
            return {'detected': False, 'message': 'No P13 data available from cycle 6 for comparison'}
        
        # Get most recent P13 from cycle 6 and first P13 from cycle 7
        cycle6_p13.sort(key=lambda x: x['tick'])
        cycle7_p13.sort(key=lambda x: x['tick'])
        
        p13_before = cycle6_p13[-1]
        p13_after = cycle7_p13[0]
        
        # Calculate transformation metrics
        phase_shift = p13_after['phase'] - p13_before['phase']
        # Normalize phase shift to [-π, π]
        phase_shift = np.arctan2(np.sin(phase_shift), np.cos(phase_shift))
        
        energy_ratio = p13_after['energy_level'] / p13_before['energy_level'] if p13_before['energy_level'] != 0 else 0
        tick_gap = p13_after['tick'] - p13_before['tick']
        
        # Check for golden ratio resonance
        phi_phase_resonance = abs(abs(phase_shift) - (1/self.phi)) < 0.1
        phi_energy_resonance = abs(energy_ratio - self.phi) < 0.2
        
        # Create transformation record
        transformation = {
            'detected': True,
            'from_tick': p13_before['tick'],
            'to_tick': p13_after['tick'],
            'tick_gap': tick_gap,
            'phase_shift': phase_shift,
            'energy_ratio': energy_ratio,
            'recursion_depth_before': p13_before['recursion_depth'],
            'recursion_depth_after': p13_after['recursion_depth'],
            'phi_phase_resonance': phi_phase_resonance,
            'phi_energy_resonance': phi_energy_resonance,
            'p13_before': p13_before,
            'p13_after': p13_after
        }
        
        # Store detected transformation
        self.detected_transitions['p13_seventh_cycle'].append(transformation)
        
        if phi_phase_resonance or phi_energy_resonance:
            logger.info(f"P13 Seventh Cycle Transformation with Golden Ratio Resonance detected at tick {p13_after['tick']}")
            logger.info(f"Phase shift: {phase_shift:.4f}, Energy ratio: {energy_ratio:.4f}")
        
        return transformation
    
    def analyze_octave_transitions(self) -> List[Dict]:
        """
        Analyze transitions between octaves (recursion levels).
        
        Returns:
            List of transition dictionaries
        """
        # Get recursion transitions from tracker
        transitions = self.recursion_tracker.get_recursion_transitions()
        
        # Group transitions by recursion depth
        transitions_by_depth = defaultdict(list)
        for t in transitions:
            depth = t['recursion_depth']
            transitions_by_depth[depth].append(t)
        
        # Analyze each depth separately
        results = []
        
        for depth, depth_transitions in transitions_by_depth.items():
            # Calculate average metrics for this depth
            avg_phase_shift = np.mean([t['phase_shift'] for t in depth_transitions])
            avg_energy_ratio = np.mean([t['energy_ratio'] for t in depth_transitions])
            
            # Count phi resonances
            phi_phase_count = sum(1 for t in depth_transitions if t['phi_phase_resonance'])
            phi_energy_count = sum(1 for t in depth_transitions if t['phi_energy_resonance'])
            
            # Create summary
            summary = {
                'recursion_depth': depth,
                'transition_count': len(depth_transitions),
                'avg_phase_shift': avg_phase_shift,
                'avg_energy_ratio': avg_energy_ratio,
                'phi_phase_resonance_count': phi_phase_count,
                'phi_energy_resonance_count': phi_energy_count,
                'phi_resonance_percentage': (phi_phase_count + phi_energy_count) / (2 * len(depth_transitions)) if depth_transitions else 0,
                'transitions': depth_transitions
            }
            
            results.append(summary)
            self.detected_transitions['octave_transitions'].append(summary)
        
        results.sort(key=lambda x: x['recursion_depth'])
        return results
    
    def detect_phi_resonances(self) -> Dict:
        """
        Detect golden ratio (φ) resonances across the system.
        
        Returns:
            Dictionary with resonance metrics
        """
        # Get position history across all depths
        all_history = []
        for depth, history in self.recursion_tracker.position_history.items():
            all_history.extend(history)
        
        if not all_history:
            return {'detected': False, 'message': 'No position history data available'}
        
        # Sort by tick
        all_history.sort(key=lambda x: x['tick'])
        
        # Analyze energy ratios and phase shifts between consecutive ticks
        energy_ratios = []
        phase_shifts = []
        
        for i in range(1, len(all_history)):
            current = all_history[i]
            previous = all_history[i-1]
            
            # Only compare same position number across consecutive ticks
            if current['position_number'] != previous['position_number']:
                continue
                
            # Calculate metrics
            energy_ratio = current['energy_level'] / previous['energy_level'] if previous['energy_level'] != 0 else 0
            phase_shift = current['phase'] - previous['phase']
            # Normalize phase shift to [-π, π]
            phase_shift = np.arctan2(np.sin(phase_shift), np.cos(phase_shift))
            
            energy_ratios.append(energy_ratio)
            phase_shifts.append(phase_shift)
        
        # Calculate histogram of energy ratios
        energy_hist, energy_bins = np.histogram(energy_ratios, bins=50, range=(0, 3))
        energy_bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
        
        # Calculate histogram of phase shifts
        phase_hist, phase_bins = np.histogram(phase_shifts, bins=50, range=(-np.pi, np.pi))
        phase_bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
        
        # Find peaks near phi and 1/phi
        phi_energy_idx = np.argmin(np.abs(energy_bin_centers - self.phi))
        invphi_phase_idx = np.argmin(np.abs(phase_bin_centers - (1/self.phi)))
        
        # Check if we have peaks near phi and 1/phi
        phi_energy_peak = energy_hist[phi_energy_idx]
        invphi_phase_peak = phase_hist[invphi_phase_idx]
        
        # Calculate resonance strength by comparing peak height to average
        avg_energy_height = np.mean(energy_hist)
        avg_phase_height = np.mean(phase_hist)
        
        phi_energy_strength = phi_energy_peak / avg_energy_height if avg_energy_height > 0 else 0
        invphi_phase_strength = invphi_phase_peak / avg_phase_height if avg_phase_height > 0 else 0
        
        # Create resonance record
        resonance = {
            'detected': phi_energy_strength > 1.5 or invphi_phase_strength > 1.5,
            'phi_energy_strength': phi_energy_strength,
            'invphi_phase_strength': invphi_phase_strength,
            'phi_energy_value': energy_bin_centers[phi_energy_idx],
            'invphi_phase_value': phase_bin_centers[invphi_phase_idx],
            'energy_histogram': {
                'counts': energy_hist.tolist(),
                'bins': energy_bin_centers.tolist()
            },
            'phase_histogram': {
                'counts': phase_hist.tolist(),
                'bins': phase_bin_centers.tolist()
            }
        }
        
        # Store resonance data
        self.detected_transitions['phi_resonances'].append(resonance)
        
        if resonance['detected']:
            logger.info(f"Phi resonance detected - Energy strength: {phi_energy_strength:.2f}, Phase strength: {invphi_phase_strength:.2f}")
        
        return resonance 