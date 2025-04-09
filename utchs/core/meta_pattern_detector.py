"""
Meta-pattern detector module for the UTCHS framework.

This module implements the MetaPatternDetector class, which identifies and analyzes
recursive meta-patterns in the UTCHS system, focusing on the 3-6-9 pattern that emerges
across multiple scales with cycle 6 as a key transition point.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict
from scipy.stats import entropy
from scipy.signal import correlate

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class MetaPatternDetector:
    """
    Detects and analyzes recursive meta-patterns in the UTCHS framework.
    
    This class focuses on identifying the recursive 3-6-9 pattern that emerges at cycle 6,
    where cycle 6 becomes a "meta-position 3" in a higher-order pattern that includes
    cycles 9 (meta-position 6) and 12 (meta-position 9).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the meta-pattern detector.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        
        # Detection thresholds
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.phase_coherence_threshold = self.config.get('phase_coherence_threshold', 0.6)
        self.energy_pattern_threshold = self.config.get('energy_pattern_threshold', 0.65)
        
        # Golden ratio (φ) for resonance detection
        self.phi = (1 + np.sqrt(5)) / 2
        
        # Storage for detected patterns
        self.detected_meta_patterns = {
            'cycle6_patterns': [],  # Meta-position 3
            'cycle9_patterns': [],  # Meta-position 6
            'cycle12_patterns': [],  # Meta-position 9
            'cross_scale_correlations': {},
            'meta_cycles_detected': 0
        }
        
        logger.info("MetaPatternDetector initialized")
    
    def detect_meta_patterns(self, position_history: Dict[int, List[Dict]], config: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Detect meta-patterns in position history data.
        
        Args:
            position_history: Dictionary of position history by recursion depth
            config: Configuration dictionary (optional)
            
        Returns:
            Dictionary with meta-pattern metrics
        """
        config = config or self.config
        
        # Extract cycle 6 positions (meta-position 3)
        cycle6_positions = self._extract_cycle_positions(position_history, 6)
        
        # Extract original position 3 data for comparison
        position3_data = self._extract_position_data(position_history, 3)
        
        # Calculate correlations between position 3 and cycle 6
        meta3_correlation = self._calculate_meta_correlation(position3_data, cycle6_positions)
        
        # Extract cycle 9 positions (meta-position 6)
        cycle9_positions = self._extract_cycle_positions(position_history, 9)
        
        # Extract original position 6 data for comparison
        position6_data = self._extract_position_data(position_history, 6)
        
        # Calculate correlations between position 6 and cycle 9
        meta6_correlation = self._calculate_meta_correlation(position6_data, cycle9_positions)
        
        # Extract cycle 12 positions (meta-position 9)
        cycle12_positions = self._extract_cycle_positions(position_history, 12)
        
        # Extract original position 9 data for comparison
        position9_data = self._extract_position_data(position_history, 9)
        
        # Calculate correlations between position 9 and cycle 12
        meta9_correlation = self._calculate_meta_correlation(position9_data, cycle12_positions)
        
        # Calculate overall meta-pattern strength
        meta_cycle_strength = self._calculate_meta_cycle_strength(
            meta3_correlation, meta6_correlation, meta9_correlation
        )
        
        # Check if we have a valid meta-pattern
        has_meta_pattern = meta_cycle_strength > config.get('meta_pattern_threshold', 0.7)
        
        # Store results
        result = {
            'detected': has_meta_pattern,
            'meta_cycle_strength': meta_cycle_strength,
            'position3_cycle6_correlation': meta3_correlation,
            'position6_cycle9_correlation': meta6_correlation,
            'position9_cycle12_correlation': meta9_correlation,
            'cycle6_meta_position3_data': {
                'count': len(cycle6_positions),
                'phase_coherence': self._calculate_phase_coherence(cycle6_positions),
                'energy_pattern': self._extract_energy_pattern(cycle6_positions)
            },
            'cycle9_meta_position6_data': {
                'count': len(cycle9_positions),
                'phase_coherence': self._calculate_phase_coherence(cycle9_positions),
                'energy_pattern': self._extract_energy_pattern(cycle9_positions)
            },
            'cycle12_meta_position9_data': {
                'count': len(cycle12_positions),
                'phase_coherence': self._calculate_phase_coherence(cycle12_positions),
                'energy_pattern': self._extract_energy_pattern(cycle12_positions)
            }
        }
        
        # Store detected pattern for later analysis
        if has_meta_pattern:
            self.detected_meta_patterns['cycle6_patterns'].append(result['cycle6_meta_position3_data'])
            self.detected_meta_patterns['cycle9_patterns'].append(result['cycle9_meta_position6_data'])
            self.detected_meta_patterns['cycle12_patterns'].append(result['cycle12_meta_position9_data'])
            self.detected_meta_patterns['cross_scale_correlations'][len(self.detected_meta_patterns['cycle6_patterns'])] = {
                'meta3': meta3_correlation,
                'meta6': meta6_correlation,
                'meta9': meta9_correlation
            }
            self.detected_meta_patterns['meta_cycles_detected'] += 1
            
            logger.info(f"Detected meta-pattern at cycle 6 with strength {meta_cycle_strength:.4f}")
        
        return result
    
    def analyze_cycle6_meta_transition(self, position_history: Dict[int, List[Dict]]) -> Dict:
        """
        Perform detailed analysis of the cycle 6 meta-transition.
        
        Args:
            position_history: Dictionary of position history by recursion depth
            
        Returns:
            Dictionary with detailed transition metrics
        """
        # Extract cycle 6 data
        cycle6_positions = self._extract_cycle_positions(position_history, 6)
        
        if not cycle6_positions:
            return {'detected': False, 'message': 'No cycle 6 data available'}
        
        # Sort by tick
        cycle6_positions.sort(key=lambda x: x['tick'])
        
        # Calculate phase shifts between consecutive ticks
        phase_shifts = []
        for i in range(1, len(cycle6_positions)):
            shift = cycle6_positions[i]['phase'] - cycle6_positions[i-1]['phase']
            # Normalize to [-π, π]
            phase_shifts.append(np.arctan2(np.sin(shift), np.cos(shift)))
        
        # Detect resonance with position 3
        position3_data = self._extract_position_data(position_history, 3)
        position3_resonance = self._detect_position_resonance(cycle6_positions, position3_data)
        
        # Analyze energy evolution pattern
        energy_values = [p['energy_level'] for p in cycle6_positions]
        energy_pattern = self._analyze_energy_evolution(energy_values)
        
        # Check for phi resonance in energy ratios
        phi_resonance = self._detect_phi_resonance(energy_values)
        
        # Calculate cyclic coherence
        cyclic_coherence = self._calculate_cyclic_coherence(phase_shifts)
        
        # Check for characteristic meta-pattern emergence signature
        meta_signature = self._detect_meta_signature(
            phase_shifts, position3_resonance, energy_pattern, phi_resonance, cyclic_coherence
        )
        
        result = {
            'detected': meta_signature['detected'],
            'confidence': meta_signature['confidence'],
            'phase_shift_pattern': phase_shifts,
            'position3_resonance': position3_resonance,
            'energy_pattern': energy_pattern,
            'phi_resonance': phi_resonance,
            'cyclic_coherence': cyclic_coherence,
            'meta_signature': meta_signature,
            'tick_range': (cycle6_positions[0]['tick'], cycle6_positions[-1]['tick'])
        }
        
        if result['detected']:
            logger.info(f"Meta-transition detected at cycle 6 with confidence {result['confidence']:.4f}")
            logger.info(f"Ticks: {result['tick_range'][0]} to {result['tick_range'][1]}")
        
        return result
    
    def predict_metacycle_evolution(self, current_order: int) -> Dict:
        """
        Predict the evolution of meta-patterns into higher orders.
        
        Args:
            current_order: Current meta-pattern order observed
            
        Returns:
            Prediction dictionary
        """
        # This is a placeholder for future implementation
        # Will predict emergence of higher-order patterns based on current observations
        next_meta3_cycle = 3 * (2 ** current_order)
        next_meta6_cycle = 6 * (2 ** current_order)
        next_meta9_cycle = 9 * (2 ** current_order)
        
        return {
            'next_order': current_order + 1,
            'predicted_cycles': {
                'meta3': next_meta3_cycle,
                'meta6': next_meta6_cycle,
                'meta9': next_meta9_cycle
            },
            'confidence': 0.8 / (current_order + 1)  # Confidence decreases with higher orders
        }
    
    def _extract_cycle_positions(self, position_history: Dict[int, List[Dict]], cycle: int) -> List[Dict]:
        """
        Extract positions from a specific cycle.
        
        Args:
            position_history: Dictionary of position history by recursion depth
            cycle: Cycle number to extract
            
        Returns:
            List of position data dictionaries for the specified cycle
        """
        cycle_positions = []
        
        for depth, history in position_history.items():
            for data in history:
                if data['absolute_position']['cycle'] == cycle:
                    cycle_positions.append(data)
        
        return cycle_positions
    
    def _extract_position_data(self, position_history: Dict[int, List[Dict]], position: int) -> List[Dict]:
        """
        Extract data for a specific position number.
        
        Args:
            position_history: Dictionary of position history by recursion depth
            position: Position number to extract (1-13)
            
        Returns:
            List of position data dictionaries for the specified position
        """
        position_data = []
        
        for depth, history in position_history.items():
            for data in history:
                if data['position_number'] == position:
                    position_data.append(data)
        
        return position_data
    
    def _calculate_meta_correlation(self, position_data: List[Dict], cycle_data: List[Dict]) -> float:
        """
        Calculate correlation between position data and cycle data.
        
        Args:
            position_data: List of position data dictionaries
            cycle_data: List of cycle data dictionaries
            
        Returns:
            Correlation value
        """
        # Extract phase values
        if not position_data or not cycle_data:
            return 0.0
        
        # Sort by tick
        position_data.sort(key=lambda x: x['tick'])
        cycle_data.sort(key=lambda x: x['tick'])
        
        # Get minimum common length
        min_len = min(len(position_data), len(cycle_data))
        if min_len < 5:  # Need at least 5 points for meaningful correlation
            return 0.0
        
        # Sample a reasonable number of points
        sample_size = min(min_len, 100)  # Cap at 100 points to avoid excessive computation
        
        # Create evenly spaced indices
        indices = np.linspace(0, min_len - 1, sample_size, dtype=int)
        
        # Extract phase and energy values
        try:
            position_phases = np.array([position_data[i]['phase'] for i in indices])
            cycle_phases = np.array([cycle_data[i]['phase'] for i in indices])
            
            position_energies = np.array([position_data[i]['energy_level'] for i in indices])
            cycle_energies = np.array([cycle_data[i]['energy_level'] for i in indices])
            
            # Compute correlations with error handling
            phase_corr = self._safe_correlation(position_phases, cycle_phases)
            energy_corr = self._safe_correlation(position_energies, cycle_energies)
            
            # Combine correlations (weight phase correlation more heavily)
            combined_corr = 0.7 * phase_corr + 0.3 * energy_corr
            
            return float(combined_corr)
        except (IndexError, ValueError) as e:
            logger.warning(f"Error calculating meta correlation: {e}")
            return 0.0
    
    def _safe_correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate correlation with safe error handling.
        
        Args:
            a: First array
            b: Second array
            
        Returns:
            Correlation coefficient
        """
        try:
            # Check if both arrays have variation
            if np.std(a) > 0 and np.std(b) > 0:
                corr = np.corrcoef(a, b)[0, 1]
                # Handle NaN values
                if np.isnan(corr):
                    return 0.0
                return corr
            else:
                # If either array has no variation, check if they're both constant
                if np.std(a) == 0 and np.std(b) == 0:
                    # If both are constant with the same value, perfect correlation
                    if np.mean(a) == np.mean(b):
                        return 1.0
                    # If both constant but different values, no correlation
                    else:
                        return 0.0
                else:
                    # One has variation, one doesn't - no meaningful correlation
                    return 0.0
        except Exception as e:
            logger.warning(f"Correlation calculation error: {e}")
            return 0.0
    
    def _calculate_meta_cycle_strength(self, meta3_corr: float, meta6_corr: float, meta9_corr: float) -> float:
        """
        Calculate overall meta-cycle strength based on individual correlations.
        
        Args:
            meta3_corr: Correlation between position 3 and cycle 6
            meta6_corr: Correlation between position 6 and cycle 9
            meta9_corr: Correlation between position 9 and cycle 12
            
        Returns:
            Meta-cycle strength value
        """
        # We weigh meta3 (cycle 6) correlation most heavily as it's the foundation
        # of the meta-pattern
        weights = [0.5, 0.3, 0.2]
        
        # Weighted average
        strength = weights[0] * meta3_corr + weights[1] * meta6_corr + weights[2] * meta9_corr
        
        # Boost strength if all correlations are above threshold
        if (meta3_corr > self.correlation_threshold and 
            meta6_corr > self.correlation_threshold and 
            meta9_corr > self.correlation_threshold):
            strength *= 1.2
            
        # Cap at 1.0
        return min(float(strength), 1.0)
    
    def _calculate_phase_coherence(self, positions: List[Dict]) -> float:
        """
        Calculate phase coherence for a set of positions.
        
        Args:
            positions: List of position data dictionaries
            
        Returns:
            Phase coherence value
        """
        if not positions:
            return 0.0
        
        # Extract phases
        phases = [p['phase'] for p in positions]
        
        # Convert to complex numbers on the unit circle
        complex_phases = np.exp(1j * np.array(phases))
        
        # Calculate mean vector
        mean_vector = np.mean(complex_phases)
        
        # Coherence is the length of the mean vector
        coherence = np.abs(mean_vector)
        
        return float(coherence)
    
    def _extract_energy_pattern(self, positions: List[Dict]) -> List[float]:
        """
        Extract energy pattern from positions.
        
        Args:
            positions: List of position data dictionaries
            
        Returns:
            List of energy values
        """
        if not positions:
            return []
        
        # Sort by tick
        positions.sort(key=lambda x: x['tick'])
        
        # Extract energy values
        energy_values = [p['energy_level'] for p in positions]
        
        return energy_values
    
    def _detect_position_resonance(self, cycle_positions: List[Dict], position_data: List[Dict]) -> Dict:
        """
        Detect resonance between cycle positions and a specific position number.
        
        Args:
            cycle_positions: List of position data dictionaries for a cycle
            position_data: List of position data dictionaries for a specific position
            
        Returns:
            Dictionary with resonance metrics
        """
        if not cycle_positions or not position_data:
            return {'detected': False, 'strength': 0.0}
        
        # Extract phases and energy levels
        cycle_phases = np.array([p['phase'] for p in cycle_positions])
        position_phases = np.array([p['phase'] for p in position_data])
        
        cycle_energies = np.array([p['energy_level'] for p in cycle_positions])
        position_energies = np.array([p['energy_level'] for p in position_data])
        
        # Calculate phase coherence
        cycle_phase_coherence = self._calculate_phase_coherence(cycle_positions)
        position_phase_coherence = self._calculate_phase_coherence(position_data)
        
        # Calculate energy pattern similarity if we have enough data
        energy_similarity = 0.0
        if len(cycle_energies) > 5 and len(position_energies) > 5:
            # Normalize energy arrays to same length
            min_len = min(len(cycle_energies), len(position_energies))
            sample_indices1 = np.linspace(0, len(cycle_energies) - 1, min_len, dtype=int)
            sample_indices2 = np.linspace(0, len(position_energies) - 1, min_len, dtype=int)
            
            sampled_cycle_energies = cycle_energies[sample_indices1]
            sampled_position_energies = position_energies[sample_indices2]
            
            # Normalize
            if np.std(sampled_cycle_energies) > 0:
                sampled_cycle_energies = (sampled_cycle_energies - np.mean(sampled_cycle_energies)) / np.std(sampled_cycle_energies)
            if np.std(sampled_position_energies) > 0:
                sampled_position_energies = (sampled_position_energies - np.mean(sampled_position_energies)) / np.std(sampled_position_energies)
            
            # Calculate correlation
            energy_similarity = self._safe_correlation(sampled_cycle_energies, sampled_position_energies)
        
        # Calculate overall resonance strength
        resonance_strength = 0.4 * cycle_phase_coherence + 0.4 * position_phase_coherence + 0.2 * energy_similarity
        
        return {
            'detected': resonance_strength > self.correlation_threshold,
            'strength': float(resonance_strength),
            'cycle_phase_coherence': float(cycle_phase_coherence),
            'position_phase_coherence': float(position_phase_coherence),
            'energy_similarity': float(energy_similarity)
        }
    
    def _analyze_energy_evolution(self, energy_values: List[float]) -> Dict:
        """
        Analyze the evolution of energy values.
        
        Args:
            energy_values: List of energy values
            
        Returns:
            Dictionary with energy pattern metrics
        """
        if len(energy_values) < 5:
            return {'pattern_detected': False}
        
        # Convert to numpy array
        energy_array = np.array(energy_values)
        
        # Calculate rate of change
        derivatives = np.diff(energy_array)
        
        # Check for oscillatory pattern (alternating signs)
        sign_changes = np.sum(np.diff(np.signbit(derivatives)) != 0)
        oscillation_score = sign_changes / (len(derivatives) - 1) if len(derivatives) > 1 else 0
        
        # Check for growth pattern
        growth_score = np.sum(derivatives > 0) / len(derivatives)
        
        # Check for phi ratio in consecutive values
        phi_ratios = []
        for i in range(1, len(energy_array)):
            if energy_array[i-1] != 0:
                ratio = energy_array[i] / energy_array[i-1]
                phi_ratios.append(abs(ratio - self.phi))
        
        phi_score = np.mean([1.0 - min(r, 1.0) for r in phi_ratios]) if phi_ratios else 0
        
        # Determine dominant pattern
        pattern_scores = {
            'oscillatory': oscillation_score,
            'growth': growth_score,
            'phi_scaled': phi_score
        }
        
        dominant_pattern = max(pattern_scores, key=pattern_scores.get)
        dominant_score = pattern_scores[dominant_pattern]
        
        return {
            'pattern_detected': dominant_score > 0.6,
            'dominant_pattern': dominant_pattern,
            'dominant_score': float(dominant_score),
            'oscillation_score': float(oscillation_score),
            'growth_score': float(growth_score),
            'phi_score': float(phi_score)
        }
    
    def _detect_phi_resonance(self, values: List[float]) -> Dict:
        """
        Detect golden ratio (φ) resonance in a series of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary with phi resonance metrics
        """
        if len(values) < 5:
            return {'detected': False, 'strength': 0.0}
        
        # Calculate ratios between consecutive values
        ratios = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                ratios.append(values[i] / values[i-1])
        
        if not ratios:
            return {'detected': False, 'strength': 0.0}
        
        # Calculate how close ratios are to phi
        phi_deviations = [abs(r - self.phi) for r in ratios]
        mean_deviation = np.mean(phi_deviations)
        
        # Higher score means closer to phi
        phi_score = max(0, 1.0 - mean_deviation / self.phi)
        
        return {
            'detected': phi_score > 0.7,
            'strength': float(phi_score),
            'mean_deviation': float(mean_deviation),
            'ratio_count': len(ratios)
        }
    
    def _calculate_cyclic_coherence(self, phase_shifts: List[float]) -> float:
        """
        Calculate cyclic coherence from phase shifts.
        
        Args:
            phase_shifts: List of phase shift values
            
        Returns:
            Cyclic coherence value
        """
        if len(phase_shifts) < 5:
            return 0.0
        
        # Convert to numpy array
        shifts = np.array(phase_shifts)
        
        # Calculate autocorrelation
        autocorr = correlate(shifts, shifts, mode='full')
        
        # Normalize
        autocorr = autocorr[len(shifts)-1:] / autocorr[len(shifts)-1]
        
        # Look for periodic patterns (sampling a few lags)
        max_lag = min(len(autocorr) - 1, 10)
        periodic_score = np.max(autocorr[1:max_lag+1])
        
        return float(periodic_score)
    
    def _detect_meta_signature(self, 
                               phase_shifts: List[float], 
                               position_resonance: Dict, 
                               energy_pattern: Dict,
                               phi_resonance: Dict,
                               cyclic_coherence: float) -> Dict:
        """
        Detect meta-pattern signature from various metrics.
        
        Args:
            phase_shifts: List of phase shift values
            position_resonance: Position resonance metrics
            energy_pattern: Energy pattern metrics
            phi_resonance: Phi resonance metrics
            cyclic_coherence: Cyclic coherence value
            
        Returns:
            Dictionary with meta-signature metrics
        """
        # Calculate component scores
        phase_score = 0.0
        if phase_shifts:
            phase_score = cyclic_coherence
        
        resonance_score = position_resonance.get('strength', 0.0)
        energy_score = energy_pattern.get('dominant_score', 0.0) if energy_pattern.get('pattern_detected', False) else 0.0
        phi_score = phi_resonance.get('strength', 0.0)
        
        # Weight the components
        weights = [0.3, 0.3, 0.2, 0.2]
        combined_score = (
            weights[0] * phase_score +
            weights[1] * resonance_score +
            weights[2] * energy_score +
            weights[3] * phi_score
        )
        
        # Determine confidence
        confidence = combined_score
        
        # Boost confidence if multiple strong indicators are present
        strong_indicators = 0
        if phase_score > 0.7: strong_indicators += 1
        if resonance_score > 0.7: strong_indicators += 1
        if energy_score > 0.7: strong_indicators += 1
        if phi_score > 0.7: strong_indicators += 1
        
        if strong_indicators >= 3:
            confidence *= 1.2
        
        confidence = min(float(confidence), 1.0)
        
        return {
            'detected': confidence > 0.65,
            'confidence': confidence,
            'component_scores': {
                'phase_score': float(phase_score),
                'resonance_score': float(resonance_score),
                'energy_score': float(energy_score),
                'phi_score': float(phi_score)
            },
            'strong_indicators': strong_indicators
        } 