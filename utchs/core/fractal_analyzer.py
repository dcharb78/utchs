"""
Fractal analyzer module for the UTCHS framework.

This module implements the FractalAnalyzer class, which quantifies self-similarity
and fractal properties across recursive levels in the UTCHS framework.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict
from scipy.stats import entropy

from ..utils.logging_config import get_logger
from .recursion_tracker import RecursionTracker
from ..mathematics.mobius import MobiusTransformation
from ..mathematics.recursion_scaling import get_correction_factor

logger = get_logger(__name__)

class FractalAnalyzer:
    """
    Analyzes self-similarity and fractal properties across recursive levels.
    
    This class focuses on quantifying fractal properties in the UTCHS framework,
    particularly how patterns repeat across different recursive levels (octaves)
    with φ-scaled dimensions.
    """
    
    def __init__(self, recursion_tracker: RecursionTracker, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the fractal analyzer.
        
        Args:
            recursion_tracker: RecursionTracker instance containing position history
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.recursion_tracker = recursion_tracker
        
        # Golden ratio (φ) for resonance detection
        self.phi = (1 + np.sqrt(5)) / 2
        
        # Möbius correction parameters
        self.enable_mobius_correction = self.config.get('enable_mobius_correction', True)
        self.mobius_correction_strength = self.config.get('mobius_correction_strength', 0.7)
        
        # Analysis results
        self.fractal_metrics = {}
        
        logger.info("FractalAnalyzer initialized with Möbius correction enabled: " +
                  f"{self.enable_mobius_correction}")
        
    def analyze_recursive_patterns(self, max_depth: int = 5) -> Dict:
        """
        Analyze recursive patterns with Möbius corrections.
        
        Args:
            max_depth: Maximum recursion depth to analyze
            
        Returns:
            Dictionary with fractal pattern metrics
        """
        # Get position data by depth
        depth_data = {}
        
        for depth in range(1, max_depth + 1):
            if depth in self.recursion_tracker.position_history:
                # Extract positions for the depth
                positions = self.recursion_tracker.position_history[depth]
                
                # Apply Möbius corrections if enabled
                if self.enable_mobius_correction and depth > 1:
                    corrected_positions = self._apply_mobius_correction(positions, depth)
                else:
                    corrected_positions = positions
                
                depth_data[depth] = {
                    'positions': corrected_positions,
                    'pattern': self._extract_pattern(corrected_positions),
                    'corrections_applied': self.enable_mobius_correction and depth > 1
                }
        
        # Analyze cross-depth patterns
        result = self._analyze_cross_depth_patterns(depth_data)
        
        self.fractal_metrics['recursive_patterns'] = result
        logger.info(f"Analyzed recursive patterns across {len(depth_data)} recursion depths")
        
        return result
    
    def _apply_mobius_correction(self, positions: List[Dict], depth: int) -> List[Dict]:
        """
        Apply Möbius correction to positions at a specific recursion depth.
        
        Args:
            positions: List of position data dictionaries
            depth: Recursion depth
            
        Returns:
            Corrected position data
        """
        if not positions:
            return positions
            
        # Create a deep copy to avoid modifying original positions
        corrected = []
        
        for pos in positions:
            # Create a copy of the position data
            corrected_pos = pos.copy()
            
            # Get current spatial location and phase
            spatial_location = np.array(pos.get('spatial_location', (0, 0, 0)))
            phase = pos.get('phase', 0)
            
            # Create a system state dict for correction calculations
            system_state = {
                'recursion_depth': depth,
                'position_number': pos.get('position_number', 0),
                'position_count': len(positions)
            }
            
            # Get correction factor for this recursion depth
            correction_factor = get_correction_factor(depth, system_state)
            
            # Apply correction to spatial location
            # We keep z-coordinate unchanged and apply correction to x, y plane
            complex_pos = complex(spatial_location[0], spatial_location[1])
            
            # Create Möbius parameters that incorporate the correction factor
            # For low correction (near 1.0), this is close to identity
            # For stronger correction, this introduces subtle nonlinear effects
            strength = self.mobius_correction_strength * (1.0 - correction_factor)
            
            # Create a Möbius transformation with correction
            a = complex(1.0 - strength * 0.1, 0)
            b = complex(0, strength * 0.05)
            c = complex(0, -strength * 0.05)
            d = complex(1.0 + strength * 0.1, 0)
            
            mobius = MobiusTransformation(a, b, c, d, recursion_order=depth)
            
            # Apply transformation to position
            transformed_pos = mobius.transform_single(complex_pos)
            
            # Update spatial location with corrected values
            corrected_spatial = (transformed_pos.real, transformed_pos.imag, spatial_location[2])
            corrected_pos['spatial_location'] = corrected_spatial
            
            # Apply phase correction if phase is present
            if 'phase' in pos:
                # Phase correction is subtler - we apply slight rotations
                phase_correction = np.exp(1j * strength * np.pi / 4)
                complex_phase = np.exp(1j * phase) * phase_correction
                corrected_pos['phase'] = np.angle(complex_phase)
            
            corrected.append(corrected_pos)
        
        logger.debug(f"Applied Möbius correction to {len(positions)} positions at depth {depth}")
        return corrected
    
    def _extract_pattern(self, positions: List[Dict]) -> Dict:
        """
        Extract pattern data from positions.
        
        Args:
            positions: List of position data dictionaries
            
        Returns:
            Dictionary with pattern metrics
        """
        # Group positions by position number (1-13)
        position_groups = defaultdict(list)
        for pos in positions:
            pos_num = pos.get('position_number', 0)
            if pos_num > 0:
                position_groups[pos_num].append(pos)
        
        # Calculate pattern metrics for each position
        pattern = {}
        for pos_num, pos_list in position_groups.items():
            # Calculate average metrics
            avg_phase = self._get_average_phase(pos_list)
            avg_energy = np.mean([p.get('energy_level', 0) for p in pos_list])
            count = len(pos_list)
            
            pattern[pos_num] = {
                'phase': avg_phase,
                'energy': avg_energy,
                'count': count
            }
        
        return pattern
    
    def _get_average_phase(self, positions: List[Dict]) -> float:
        """
        Calculate average phase from a list of positions.
        
        Args:
            positions: List of position data dictionaries
            
        Returns:
            Average phase value
        """
        if not positions:
            return 0.0
        
        # Extract phases
        phases = [p.get('phase', 0) for p in positions if 'phase' in p]
        
        if not phases:
            return 0.0
        
        # Convert to unit vectors
        complex_phases = np.exp(1j * np.array(phases))
        
        # Calculate mean vector
        mean_vector = np.mean(complex_phases)
        
        # Return the angle of the mean vector
        return np.angle(mean_vector)
    
    def _analyze_cross_depth_patterns(self, depth_data: Dict[int, Dict]) -> Dict:
        """
        Analyze patterns across recursion depths.
        
        Args:
            depth_data: Dictionary of pattern data by recursion depth
            
        Returns:
            Dictionary with cross-depth pattern metrics
        """
        result = {
            'depth_data': {},
            'cross_depth_correlations': {},
            'scaling_factors': {},
            'phi_resonance': {},
            'corrections_applied': {}
        }
        
        # Process each depth
        for depth, data in depth_data.items():
            pattern = data['pattern']
            corrections_applied = data.get('corrections_applied', False)
            
            # Store basic metrics
            result['depth_data'][depth] = {
                'position_counts': {pos: data['count'] for pos, data in pattern.items()},
                'energy_pattern': {pos: data['energy'] for pos, data in pattern.items()},
                'corrections_applied': corrections_applied
            }
            
            # Record correction status
            result['corrections_applied'][depth] = corrections_applied
            
            # Calculate phi resonance for this depth
            phi_resonance = self._calculate_phi_scaling_within_depth(depth)
            result['phi_resonance'][depth] = phi_resonance
        
        # Calculate cross-depth correlations and scaling factors
        depths = sorted(depth_data.keys())
        for i in range(len(depths) - 1):
            depth1 = depths[i]
            depth2 = depths[i+1]
            
            # Calculate correlation between adjacent depths
            correlation = self._calculate_pattern_correlation(
                depth_data[depth1]['pattern'],
                depth_data[depth2]['pattern']
            )
            
            # Calculate energy scaling factors
            scaling_factors = self._calculate_energy_scaling(
                depth_data[depth1]['pattern'],
                depth_data[depth2]['pattern']
            )
            
            # Store results
            result['cross_depth_correlations'][(depth1, depth2)] = correlation
            result['scaling_factors'][(depth1, depth2)] = scaling_factors
        
        return result
    
    def _calculate_pattern_correlation(self, pattern1: Dict, pattern2: Dict) -> float:
        """
        Calculate correlation between two patterns.
        
        Args:
            pattern1, pattern2: Pattern dictionaries
            
        Returns:
            Correlation value (0.0-1.0)
        """
        # Find common positions
        common_positions = set(pattern1.keys()).intersection(set(pattern2.keys()))
        
        if not common_positions:
            return 0.0
        
        # Calculate energy correlation
        energy1 = [pattern1[pos]['energy'] for pos in common_positions]
        energy2 = [pattern2[pos]['energy'] for pos in common_positions]
        
        # Calculate phase correlation
        phase1 = [pattern1[pos]['phase'] for pos in common_positions]
        phase2 = [pattern2[pos]['phase'] for pos in common_positions]
        
        # Convert phase to complex
        complex_phase1 = np.exp(1j * np.array(phase1))
        complex_phase2 = np.exp(1j * np.array(phase2))
        
        # Calculate correlation
        energy_corr = np.corrcoef(energy1, energy2)[0, 1] if len(energy1) > 1 else 0.0
        phase_corr = np.abs(np.mean(complex_phase1 * np.conjugate(complex_phase2)))
        
        # Combine correlations (weight energy more as it's more reliable)
        combined_corr = 0.7 * energy_corr + 0.3 * phase_corr
        
        # Handle NaN
        if np.isnan(combined_corr):
            return 0.0
            
        return float(combined_corr)
    
    def _calculate_energy_scaling(self, pattern1: Dict, pattern2: Dict) -> Dict:
        """
        Calculate energy scaling factors between patterns.
        
        Args:
            pattern1, pattern2: Pattern dictionaries
            
        Returns:
            Dictionary with scaling factors
        """
        # Find common positions
        common_positions = set(pattern1.keys()).intersection(set(pattern2.keys()))
        
        scaling_factors = {}
        
        for pos in common_positions:
            energy1 = pattern1[pos]['energy']
            energy2 = pattern2[pos]['energy']
            
            if energy1 > 0:
                factor = energy2 / energy1
                scaling_factors[pos] = factor
        
        # Calculate overall scaling
        if scaling_factors:
            factors = list(scaling_factors.values())
            avg_factor = np.mean(factors)
            phi_deviation = abs(avg_factor - self.phi)
            phi_resonance = phi_deviation < 0.2
            
            return {
                'factors': scaling_factors,
                'average': float(avg_factor),
                'phi_deviation': float(phi_deviation),
                'phi_resonance': phi_resonance
            }
        
        return {'factors': {}, 'average': 0.0, 'phi_deviation': float('inf'), 'phi_resonance': False}
    
    def calculate_fractal_dimension(self) -> Dict:
        """
        Calculate fractal dimension using box-counting method.
        
        Returns:
            Dictionary with fractal dimension metrics
        """
        # We need sufficient data across multiple recursion levels
        has_sufficient_data = False
        for depth, history in self.recursion_tracker.position_history.items():
            if len(history) > 10:  # Arbitrary threshold
                has_sufficient_data = True
                break
                
        if not has_sufficient_data:
            return {'dimension': 0.0, 'error': 'Insufficient data for fractal dimension calculation'}
        
        # Get spatial locations across recursion levels
        points_by_depth = defaultdict(list)
        
        for depth, history in self.recursion_tracker.position_history.items():
            for data in history:
                points_by_depth[depth].append(data['spatial_location'])
        
        # Calculate box counts at different scales
        scales = []
        counts = []
        
        for scale_factor in [0.5, 0.25, 0.125, 0.0625]:
            # Create grid at this scale
            grid_size = 1.0 / scale_factor
            occupied_boxes = set()
            
            # Count occupied boxes for all points
            for depth, points in points_by_depth.items():
                for point in points:
                    # Map point to box coordinates at this scale
                    box_x = int(point[0] * grid_size)
                    box_y = int(point[1] * grid_size)
                    box_z = int(point[2] * grid_size)
                    
                    # Add to set of occupied boxes
                    occupied_boxes.add((box_x, box_y, box_z))
            
            # Store results
            scales.append(np.log(1.0 / scale_factor))
            counts.append(np.log(len(occupied_boxes)))
        
        # Calculate fractal dimension as slope of log-log plot
        if len(scales) > 1:
            # Use numpy's polyfit to calculate slope
            slope, intercept = np.polyfit(scales, counts, 1)
            r_squared = np.corrcoef(scales, counts)[0, 1] ** 2
            
            fractal_dimension = slope
        else:
            fractal_dimension = 0.0
            r_squared = 0.0
        
        result = {
            'dimension': fractal_dimension,
            'r_squared': r_squared,
            'scales': scales,
            'counts': counts
        }
        
        self.fractal_metrics['fractal_dimension'] = result
        
        logger.info(f"Calculated fractal dimension: {fractal_dimension:.4f} (R²={r_squared:.4f})")
        
        return result
    
    def calculate_multi_scale_entropy(self) -> Dict:
        """
        Calculate multi-scale entropy to measure complexity across scales.
        
        Returns:
            Dictionary with entropy metrics
        """
        # Get phase data for each position across recursion depths
        position_phase_by_depth = defaultdict(lambda: defaultdict(list))
        
        for depth, history in self.recursion_tracker.position_history.items():
            for data in history:
                pos_num = data['position_number']
                position_phase_by_depth[depth][pos_num].append(data['phase'])
        
        # Entropy at each scale
        entropy_by_depth = {}
        
        for depth, positions in position_phase_by_depth.items():
            if not positions:
                continue
                
            # Calculate entropy for each position at this depth
            position_entropy = {}
            
            for pos_num, phases in positions.items():
                if len(phases) < 10:  # Need enough data points
                    continue
                    
                # Use histogram to estimate probability distribution
                hist, _ = np.histogram(phases, bins=10, range=(-np.pi, np.pi), density=True)
                
                # Calculate entropy: -sum(p * log(p))
                entropy_val = 0.0
                for p in hist:
                    if p > 0:
                        entropy_val -= p * np.log(p)
                
                position_entropy[pos_num] = entropy_val
            
            # Average entropy across all positions at this depth
            if position_entropy:
                entropy_by_depth[depth] = {
                    'mean': np.mean(list(position_entropy.values())),
                    'std': np.std(list(position_entropy.values())),
                    'by_position': position_entropy
                }
        
        # Analyze entropy scaling across depths
        depths = sorted(entropy_by_depth.keys())
        depth_entropy = [entropy_by_depth[d]['mean'] for d in depths]
        
        # Store results
        result = {
            'entropy_by_depth': entropy_by_depth,
            'depths': depths,
            'mean_entropy': depth_entropy
        }
        
        self.fractal_metrics['multi_scale_entropy'] = result
        
        if depth_entropy:
            logger.info(f"Multi-scale entropy calculated across {len(depths)} recursion depths")
        
        return result
    
    def calculate_self_similarity(self) -> Dict:
        """
        Calculate self-similarity between patterns at different recursion depths.
        
        Returns:
            Dictionary with self-similarity metrics
        """
        # We need data for at least 2 recursion depths
        if len(self.recursion_tracker.position_history) < 2:
            # Use alternative method when only one recursion depth is available
            return self._calculate_single_depth_similarity()
        
        # Calculate similarity between consecutive recursion depths
        similarity_scores = []
        phi_scaling_scores = []
        
        for depth in range(self.recursion_tracker.max_recursion_depth - 1):
            if depth not in self.recursion_tracker.position_history or depth+1 not in self.recursion_tracker.position_history:
                continue
                
            # Get energy patterns for each depth
            depth_energy_pattern = self._get_depth_energy_pattern(depth)
            next_depth_energy_pattern = self._get_depth_energy_pattern(depth+1)
            
            if not depth_energy_pattern or not next_depth_energy_pattern:
                continue
            
            # Calculate correlation between patterns - with proper error handling
            try:
                # Check if both patterns have variation (standard deviation > 0)
                std1 = np.std(depth_energy_pattern)
                std2 = np.std(next_depth_energy_pattern)
                
                if std1 > 0 and std2 > 0:
                    # Standard correlation
                    correlation = np.corrcoef(depth_energy_pattern, next_depth_energy_pattern)[0, 1]
                    
                    # Replace NaN with 0 if it still occurs
                    if np.isnan(correlation):
                        correlation = 0.0
                        logger.warning("NaN encountered in correlation calculation, using 0.0 as fallback")
                else:
                    # Handle constant patterns (no variation) by comparing means
                    mean1 = np.mean(depth_energy_pattern)
                    mean2 = np.mean(next_depth_energy_pattern)
                    
                    # If means are close, patterns are similar
                    if abs(mean1 - mean2) < 0.1 * max(mean1, mean2):
                        correlation = 1.0  # High similarity for near-identical constant patterns
                    else:
                        correlation = 0.0  # Low similarity for different constant patterns
                    
                    logger.debug(f"Constant pattern detected, using mean comparison: correlation={correlation}")
            except Exception as e:
                logger.warning(f"Error calculating correlation: {e}, using 0.0 as fallback")
                correlation = 0.0
            
            # Calculate energy scaling factor with error handling
            try:
                mean1 = np.mean(depth_energy_pattern)
                mean2 = np.mean(next_depth_energy_pattern)
                
                if mean1 > 0:
                    average_scaling = mean2 / mean1
                else:
                    average_scaling = 0.0
                    logger.debug("Zero mean in depth pattern, scaling factor set to 0.0")
            except Exception as e:
                logger.warning(f"Error calculating scaling factor: {e}, using 0.0 as fallback")
                average_scaling = 0.0
            
            # Check if scaling is close to phi
            phi_scaling = abs(average_scaling - self.phi) < 0.2
            
            similarity_scores.append(correlation)
            phi_scaling_scores.append(phi_scaling)
        
        # Calculate average similarity and phi scaling percentage with error handling
        if similarity_scores:
            avg_similarity = float(np.mean(similarity_scores))
            phi_scaling_percentage = float(np.mean(phi_scaling_scores))
        else:
            avg_similarity = 0.0
            phi_scaling_percentage = 0.0
        
        result = {
            'self_similarity': avg_similarity,
            'phi_scaling_percentage': phi_scaling_percentage,
            'depth_pairs_analyzed': len(similarity_scores),
            'similarity_scores': similarity_scores,
            'phi_scaling_scores': phi_scaling_scores,
            'method': 'multi_depth'
        }
        
        self.fractal_metrics['self_similarity'] = result
        
        logger.info(f"Self-similarity: {avg_similarity:.4f}, Phi-scaling percentage: {phi_scaling_percentage:.2%}")
        
        return result
    
    def _calculate_single_depth_similarity(self) -> Dict:
        """
        Calculate self-similarity metrics when only one recursion depth is available.
        This uses alternative methods that don't require multiple recursion levels.
        
        Returns:
            Dictionary with self-similarity metrics
        """
        # Get the single available depth
        available_depths = list(self.recursion_tracker.position_history.keys())
        if not available_depths:
            return {
                'self_similarity': 0.0,
                'phi_scaling_percentage': 0.0,
                'error': 'No position history data available',
                'method': 'single_depth'
            }
        
        depth = available_depths[0]
        
        # Calculate temporal self-similarity by comparing patterns across time
        # Divide history into segments and compare patterns
        history = list(self.recursion_tracker.position_history[depth])
        
        if len(history) < 20:  # Need sufficient history for meaningful comparison
            return {
                'self_similarity': 0.0,
                'phi_scaling_percentage': 0.0,
                'error': 'Insufficient history for single-depth similarity',
                'method': 'single_depth'
            }
        
        # Split history into earlier and later segments
        mid_point = len(history) // 2
        early_segment = history[:mid_point]
        late_segment = history[mid_point:]
        
        # Create energy patterns for each segment
        early_pattern = self._get_temporal_energy_pattern(early_segment)
        late_pattern = self._get_temporal_energy_pattern(late_segment)
        
        # Calculate correlation between temporal patterns with error handling
        try:
            # Check if both patterns have variation
            std1 = np.std(early_pattern)
            std2 = np.std(late_pattern)
            
            if std1 > 0 and std2 > 0:
                correlation = np.corrcoef(early_pattern, late_pattern)[0, 1]
                
                # Replace NaN with 0 if it still occurs
                if np.isnan(correlation):
                    correlation = 0.0
                    logger.warning("NaN encountered in single-depth correlation, using 0.0 as fallback")
            else:
                # Handle constant patterns
                mean1 = np.mean(early_pattern)
                mean2 = np.mean(late_pattern)
                
                if abs(mean1 - mean2) < 0.1 * max(mean1, mean2):
                    correlation = 1.0
                else:
                    correlation = 0.0
                
                logger.debug(f"Constant pattern in single-depth analysis, correlation={correlation}")
        except Exception as e:
            logger.warning(f"Error calculating single-depth correlation: {e}, using 0.0 as fallback")
            correlation = 0.0
        
        # For phi scaling, compare with theoretical pattern
        phi_scaling_score = self._calculate_phi_scaling_within_depth(depth)
        
        result = {
            'self_similarity': float(correlation),
            'phi_scaling_percentage': phi_scaling_score,
            'segments_compared': 2,
            'early_segment_size': len(early_segment),
            'late_segment_size': len(late_segment),
            'method': 'single_depth_temporal'
        }
        
        self.fractal_metrics['self_similarity'] = result
        
        logger.info(f"Single-depth self-similarity: {correlation:.4f}, Phi-scaling: {phi_scaling_score:.2%}")
        
        return result
    
    def _get_temporal_energy_pattern(self, history_segment: List[Dict]) -> List[float]:
        """
        Create energy pattern from a segment of position history.
        
        Args:
            history_segment: List of position data dictionaries
            
        Returns:
            List of energy values by position
        """
        # Group by position number
        position_data = defaultdict(list)
        for data in history_segment:
            position_data[data['position_number']].append(data['energy_level'])
        
        # Calculate average energy for each position
        pattern = []
        for pos in range(1, 14):  # Positions 1-13
            if pos in position_data:
                avg_energy = np.mean(position_data[pos])
                pattern.append(avg_energy)
            else:
                pattern.append(0.0)
                
        return pattern
    
    def _calculate_phi_scaling_within_depth(self, depth: int) -> float:
        """
        Calculate phi scaling score within a single recursion depth.
        
        Args:
            depth: Recursion depth to analyze
            
        Returns:
            Phi scaling score between 0.0 and 1.0
        """
        # Get energy pattern for the depth
        pattern = self._get_depth_energy_pattern(depth)
        
        if not pattern or len(pattern) < 13:
            return 0.0
            
        # Check for phi-based scaling between positions
        # Several key relationships to check:
        # 1. P8/P5 (should be close to phi)
        # 2. P13/P8 (should be close to phi)
        # 3. P5/P3 (should be close to phi)
        phi_relationships = [
            (8, 5),  # P8/P5
            (13, 8), # P13/P8
            (5, 3)   # P5/P3
        ]
        
        phi_scores = []
        for pos1, pos2 in phi_relationships:
            # Adjust for 0-indexing in pattern
            idx1, idx2 = pos1 - 1, pos2 - 1
            
            # Check if we have non-zero values
            if pattern[idx2] > 0:
                ratio = pattern[idx1] / pattern[idx2]
                # Calculate how close the ratio is to phi
                phi_score = 1.0 - min(abs(ratio - self.phi) / self.phi, 1.0)
                phi_scores.append(phi_score)
        
        # Return average phi score
        return np.mean(phi_scores) if phi_scores else 0.0
    
    def _get_depth_energy_pattern(self, depth: int) -> List[float]:
        """
        Get energy pattern for a specific recursion depth.
        
        Args:
            depth: Recursion depth
            
        Returns:
            List of energy values by position
        """
        if depth not in self.recursion_tracker.position_history:
            return []
            
        # Group data by position number
        position_data = defaultdict(list)
        for data in self.recursion_tracker.position_history[depth]:
            pos_num = data['position_number']
            position_data[pos_num].append(data['energy_level'])
        
        # Calculate average energy for each position
        pattern = []
        for pos in range(1, 14):  # Positions 1-13
            if pos in position_data:
                avg_energy = np.mean(position_data[pos])
                pattern.append(avg_energy)
            else:
                pattern.append(0.0)
                
        return pattern
    
    def calculate_scaling_invariance(self) -> Dict:
        """
        Calculate scaling invariance properties across recursion depths.
        
        Returns:
            Dictionary with scaling invariance metrics
        """
        # Calculate scaling factors for each position
        scaling_factors = {}
        
        for pos in range(1, 14):  # Positions 1-13
            # Get position data across recursion levels
            pos_data_by_depth = {}
            
            for depth, history in self.recursion_tracker.position_history.items():
                pos_energy = [data['energy_level'] for data in history if data['position_number'] == pos]
                if pos_energy:
                    pos_data_by_depth[depth] = np.mean(pos_energy)
            
            # Calculate scaling factor between consecutive depths
            if len(pos_data_by_depth) > 1:
                factors = []
                depths = sorted(pos_data_by_depth.keys())
                
                for i in range(len(depths) - 1):
                    current_depth = depths[i]
                    next_depth = depths[i+1]
                    
                    if pos_data_by_depth[current_depth] > 0:
                        factor = pos_data_by_depth[next_depth] / pos_data_by_depth[current_depth]
                        factors.append(factor)
                        
                if factors:
                    scaling_factors[pos] = {
                        'mean_factor': np.mean(factors),
                        'std_factor': np.std(factors),
                        'factors': factors
                    }
        
        # If we only have one recursion depth, use alternative method
        if len(self.recursion_tracker.position_history) < 2:
            return self._calculate_single_depth_scaling_invariance()
            
        # Calculate average scaling factor across all positions
        all_factors = []
        for pos, data in scaling_factors.items():
            all_factors.extend(data['factors'])
            
        avg_factor = np.mean(all_factors) if all_factors else 0.0
        phi_deviation = abs(avg_factor - self.phi) if all_factors else float('inf')
        
        # Determine invariant positions (closest to phi scaling)
        invariant_positions = []
        for pos, data in scaling_factors.items():
            if abs(data['mean_factor'] - self.phi) < 0.2:
                invariant_positions.append(pos)
                
        result = {
            'average_scaling_factor': float(avg_factor),
            'phi_deviation': float(phi_deviation),
            'phi_resonance': phi_deviation < 0.2,
            'invariant_positions': invariant_positions,
            'position_scaling_factors': scaling_factors,
            'method': 'multi_depth'
        }
        
        self.fractal_metrics['scaling_invariance'] = result
        
        if all_factors:
            logger.info(f"Average scaling factor: {avg_factor:.4f} (deviation from φ: {phi_deviation:.4f})")
            logger.info(f"Invariant positions: {invariant_positions}")
            
        return result
        
    def _calculate_single_depth_scaling_invariance(self) -> Dict:
        """
        Calculate scaling invariance using only data from a single recursion depth.
        This examines the relationships between positions within the same depth.
        
        Returns:
            Dictionary with scaling invariance metrics
        """
        available_depths = list(self.recursion_tracker.position_history.keys())
        if not available_depths:
            return {
                'average_scaling_factor': 0.0,
                'phi_deviation': float('inf'),
                'phi_resonance': False,
                'invariant_positions': [],
                'method': 'single_depth',
                'error': 'No position history data available'
            }
            
        depth = available_depths[0]
        pattern = self._get_depth_energy_pattern(depth)
        
        if not pattern or len(pattern) < 13:
            return {
                'average_scaling_factor': 0.0,
                'phi_deviation': float('inf'),
                'phi_resonance': False,
                'invariant_positions': [],
                'method': 'single_depth',
                'error': 'Insufficient pattern data'
            }
            
        # Key Fibonacci-like position pairs to check for phi scaling
        # These are positions known to have phi relationships in UTCHS theory
        phi_pairs = [
            (13, 8), # P13/P8
            (8, 5),  # P8/P5
            (5, 3),  # P5/P3
            (3, 2),  # P3/P2
            (2, 1)   # P2/P1
        ]
        
        # Calculate scaling factors between these positions
        scaling_factors = {}
        all_factors = []
        
        for pos1, pos2 in phi_pairs:
            idx1, idx2 = pos1-1, pos2-1  # Adjust for 0-indexing
            
            if pattern[idx2] > 0:
                factor = pattern[idx1] / pattern[idx2]
                all_factors.append(factor)
                scaling_factors[f"P{pos1}/P{pos2}"] = factor
                
        # Calculate average scaling factor
        avg_factor = np.mean(all_factors) if all_factors else 0.0
        phi_deviation = abs(avg_factor - self.phi) if all_factors else float('inf')
        
        # Determine invariant positions
        invariant_positions = []
        for i in range(len(pattern)-1):
            pos = i+1  # Convert to 1-indexed position
            if pos+1 <= 13 and pattern[i+1] > 0:
                ratio = pattern[i] / pattern[i+1]
                if abs(ratio - self.phi) < 0.2:
                    invariant_positions.append(pos)
                    
        result = {
            'average_scaling_factor': float(avg_factor),
            'phi_deviation': float(phi_deviation),
            'phi_resonance': phi_deviation < 0.2,
            'invariant_positions': invariant_positions,
            'fibonacci_pair_factors': scaling_factors,
            'method': 'single_depth_positions'
        }
        
        self.fractal_metrics['scaling_invariance'] = result
        
        logger.info(f"Single-depth scaling factor: {avg_factor:.4f} (deviation from φ: {phi_deviation:.4f})")
        logger.info(f"Invariant positions within depth: {invariant_positions}")
            
        return result 