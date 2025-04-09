"""
Meta-pattern detector module for the UTCHS framework.

This module implements the MetaPatternDetector class, which identifies and analyzes
recursive meta-patterns in the UTCHS system, focusing on the 3-6-9 pattern that emerges
across multiple scales with cycle 6 as a key transition point.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
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
    cycles 9 (meta-position 6) and 12 (meta-position 9). This pattern can continue
    recursively at higher levels, creating patterns within patterns within patterns.
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
        
        # Maximum recursion order to check (how many levels of meta-patterns)
        self.max_recursion_order = self.config.get('max_recursion_order', 5)
        
        # Storage for detected patterns at different recursion levels
        self.detected_meta_patterns = defaultdict(lambda: {
            'cycle_patterns': defaultdict(list),   # Meta-position patterns by cycle
            'cross_scale_correlations': {},        # Correlations between levels
            'patterns_detected': 0                 # Count of patterns at this level
        })
        
        # Storage for propagation analysis
        self.pattern_propagation = defaultdict(list)
        
        logger.info("MetaPatternDetector initialized with max recursion order: " + 
                   f"{self.max_recursion_order}")
    
    def detect_meta_patterns(self, 
                             position_history: Dict[int, List[Dict]], 
                             recursion_order: int = 2,
                             config: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Detect meta-patterns in position history data at a specific recursion order.
        
        Args:
            position_history: Dictionary of position history by recursion depth
            recursion_order: Order of meta-pattern to detect (2=cycle 6 as meta-3, etc.)
            config: Configuration dictionary (optional)
            
        Returns:
            Dictionary with meta-pattern metrics
        """
        config = config or self.config
        
        # Calculate meta-positions for this recursion order
        meta3_cycle = self._calculate_meta_position_cycle(3, recursion_order)
        meta6_cycle = self._calculate_meta_position_cycle(6, recursion_order)
        meta9_cycle = self._calculate_meta_position_cycle(9, recursion_order)
        
        logger.debug(f"Detecting meta-patterns at recursion order {recursion_order} " +
                    f"with cycles: {meta3_cycle}, {meta6_cycle}, {meta9_cycle}")
        
        # Get positions from corresponding cycles
        meta3_positions = self._extract_cycle_positions(position_history, meta3_cycle)
        meta6_positions = self._extract_cycle_positions(position_history, meta6_cycle)
        meta9_positions = self._extract_cycle_positions(position_history, meta9_cycle)
        
        # Get original or lower meta-positions for comparison
        original_position3 = self._get_meta_position_data(position_history, 3, recursion_order-1)
        original_position6 = self._get_meta_position_data(position_history, 6, recursion_order-1)
        original_position9 = self._get_meta_position_data(position_history, 9, recursion_order-1)
        
        # Calculate correlations between meta-positions and their "originals"
        meta3_correlation = self._calculate_meta_correlation(original_position3, meta3_positions)
        meta6_correlation = self._calculate_meta_correlation(original_position6, meta6_positions)
        meta9_correlation = self._calculate_meta_correlation(original_position9, meta9_positions)
        
        # Calculate overall meta-pattern strength
        meta_cycle_strength = self._calculate_meta_cycle_strength(
            meta3_correlation, meta6_correlation, meta9_correlation
        )
        
        # Check if we have a valid meta-pattern at this recursion order
        has_meta_pattern = meta_cycle_strength > config.get('meta_pattern_threshold', 0.7)
        
        # Store results
        result = {
            'detected': has_meta_pattern,
            'recursion_order': recursion_order,
            'meta_cycle_strength': meta_cycle_strength,
            'meta3_cycle': meta3_cycle,
            'meta6_cycle': meta6_cycle,
            'meta9_cycle': meta9_cycle,
            'meta3_correlation': meta3_correlation,
            'meta6_correlation': meta6_correlation,
            'meta9_correlation': meta9_correlation,
            'meta3_data': {
                'count': len(meta3_positions),
                'phase_coherence': self._calculate_phase_coherence(meta3_positions),
                'energy_pattern': self._extract_energy_pattern(meta3_positions)
            },
            'meta6_data': {
                'count': len(meta6_positions),
                'phase_coherence': self._calculate_phase_coherence(meta6_positions),
                'energy_pattern': self._extract_energy_pattern(meta6_positions)
            },
            'meta9_data': {
                'count': len(meta9_positions),
                'phase_coherence': self._calculate_phase_coherence(meta9_positions),
                'energy_pattern': self._extract_energy_pattern(meta9_positions)
            }
        }
        
        # Store detected pattern for later analysis
        if has_meta_pattern:
            self.detected_meta_patterns[recursion_order]['cycle_patterns'][meta3_cycle].append(result['meta3_data'])
            self.detected_meta_patterns[recursion_order]['cycle_patterns'][meta6_cycle].append(result['meta6_data'])
            self.detected_meta_patterns[recursion_order]['cycle_patterns'][meta9_cycle].append(result['meta9_data'])
            
            pattern_idx = self.detected_meta_patterns[recursion_order]['patterns_detected']
            self.detected_meta_patterns[recursion_order]['cross_scale_correlations'][pattern_idx] = {
                'meta3': meta3_correlation,
                'meta6': meta6_correlation,
                'meta9': meta9_correlation
            }
            self.detected_meta_patterns[recursion_order]['patterns_detected'] += 1
            
            logger.info(f"Detected meta-pattern at recursion order {recursion_order} " +
                       f"with strength {meta_cycle_strength:.4f}")
        
        return result
    
    def detect_dimensional_systems(self, 
                             position_history: Dict[int, List[Dict]], 
                             recursion_order: int = 2,
                             config: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Detect complete 13D systems generated at a specific recursion order.
        
        This method identifies the emergence of entire new 13D systems generated
        by the 3-6-9 pattern at different recursion orders. Each new system contains
        a complete set of 13 positions functioning as a coherent dimensional framework.
        
        Args:
            position_history: Dictionary of position history by recursion depth
            recursion_order: Order of dimensional system to detect 
                             (1=base system, 2=first nested system at cycle 6, etc.)
            config: Configuration dictionary (optional)
            
        Returns:
            Dictionary with metrics about the detected dimensional system
        """
        config = config or self.config
        
        # Calculate meta-positions for this recursion order
        meta3_cycle = self._calculate_meta_position_cycle(3, recursion_order)
        meta6_cycle = self._calculate_meta_position_cycle(6, recursion_order)
        meta9_cycle = self._calculate_meta_position_cycle(9, recursion_order)
        
        logger.debug(f"Detecting meta-patterns at recursion order {recursion_order} " +
                    f"with cycles: {meta3_cycle}, {meta6_cycle}, {meta9_cycle}")
        
        # Get positions from corresponding cycles
        meta3_positions = self._extract_cycle_positions(position_history, meta3_cycle)
        meta6_positions = self._extract_cycle_positions(position_history, meta6_cycle)
        meta9_positions = self._extract_cycle_positions(position_history, meta9_cycle)
        
        # Get original or lower meta-positions for comparison
        original_position3 = self._get_meta_position_data(position_history, 3, recursion_order-1)
        original_position6 = self._get_meta_position_data(position_history, 6, recursion_order-1)
        original_position9 = self._get_meta_position_data(position_history, 9, recursion_order-1)
        
        # Calculate correlations between meta-positions and their "originals"
        meta3_correlation = self._calculate_meta_correlation(original_position3, meta3_positions)
        meta6_correlation = self._calculate_meta_correlation(original_position6, meta6_positions)
        meta9_correlation = self._calculate_meta_correlation(original_position9, meta9_positions)
        
        # Calculate overall meta-pattern strength
        meta_cycle_strength = self._calculate_meta_cycle_strength(
            meta3_correlation, meta6_correlation, meta9_correlation
        )
        
        # Check if we have a valid meta-pattern at this recursion order
        has_meta_pattern = meta_cycle_strength > config.get('meta_pattern_threshold', 0.7)
        
        # Store results
        result = {
            'detected': has_meta_pattern,
            'recursion_order': recursion_order,
            'meta_cycle_strength': meta_cycle_strength,
            'meta3_cycle': meta3_cycle,
            'meta6_cycle': meta6_cycle,
            'meta9_cycle': meta9_cycle,
            'meta3_correlation': meta3_correlation,
            'meta6_correlation': meta6_correlation,
            'meta9_correlation': meta9_correlation,
            'meta3_data': {
                'count': len(meta3_positions),
                'phase_coherence': self._calculate_phase_coherence(meta3_positions),
                'energy_pattern': self._extract_energy_pattern(meta3_positions)
            },
            'meta6_data': {
                'count': len(meta6_positions),
                'phase_coherence': self._calculate_phase_coherence(meta6_positions),
                'energy_pattern': self._extract_energy_pattern(meta6_positions)
            },
            'meta9_data': {
                'count': len(meta9_positions),
                'phase_coherence': self._calculate_phase_coherence(meta9_positions),
                'energy_pattern': self._extract_energy_pattern(meta9_positions)
            }
        }
        
        # Store detected pattern for later analysis
        if has_meta_pattern:
            self.detected_meta_patterns[recursion_order]['cycle_patterns'][meta3_cycle].append(result['meta3_data'])
            self.detected_meta_patterns[recursion_order]['cycle_patterns'][meta6_cycle].append(result['meta6_data'])
            self.detected_meta_patterns[recursion_order]['cycle_patterns'][meta9_cycle].append(result['meta9_data'])
            
            pattern_idx = self.detected_meta_patterns[recursion_order]['patterns_detected']
            self.detected_meta_patterns[recursion_order]['cross_scale_correlations'][pattern_idx] = {
                'meta3': meta3_correlation,
                'meta6': meta6_correlation,
                'meta9': meta9_correlation
            }
            self.detected_meta_patterns[recursion_order]['patterns_detected'] += 1
            
            logger.info(f"Detected meta-pattern at recursion order {recursion_order} " +
                       f"with strength {meta_cycle_strength:.4f}")
        
        return result
    
    def detect_all_meta_patterns(self, position_history: Dict[int, List[Dict]]) -> Dict[int, Dict]:
        """
        Detect meta-patterns at all recursion orders up to max_recursion_order.
        
        Args:
            position_history: Dictionary of position history by recursion depth
            
        Returns:
            Dictionary of meta-pattern results by recursion order
        """
        all_results = {}
        
        # First, detect the base-level 3-6-9 pattern (order 1)
        base_pattern = self._analyze_base_pattern(position_history)
        all_results[1] = base_pattern
        
        # Then detect higher-order meta-patterns
        for order in range(2, self.max_recursion_order + 1):
            result = self.detect_meta_patterns(position_history, order)
            all_results[order] = result
            
            # If we don't detect a pattern at this level, unlikely to find at higher levels
            if not result['detected'] and self.config.get('stop_at_first_missing', True):
                logger.info(f"No meta-pattern detected at order {order}, stopping detection")
                break
        
        # Analyze propagation between levels
        self._analyze_pattern_propagation(all_results)
        
        return all_results
    
    def analyze_cycle_meta_transition(self, 
                                     position_history: Dict[int, List[Dict]],
                                     cycle: int,
                                     recursion_order: int = 2) -> Dict:
        """
        Perform detailed analysis of a specific meta-transition at any cycle.
        
        Args:
            position_history: Dictionary of position history by recursion depth
            cycle: Cycle number to analyze
            recursion_order: Recursion order to which this cycle belongs
            
        Returns:
            Dictionary with detailed transition metrics
        """
        # Extract cycle data
        cycle_positions = self._extract_cycle_positions(position_history, cycle)
        
        if not cycle_positions:
            return {'detected': False, 'message': f'No cycle {cycle} data available'}
        
        # Sort by tick
        cycle_positions.sort(key=lambda x: x['tick'])
        
        # Calculate phase shifts between consecutive ticks
        phase_shifts = []
        for i in range(1, len(cycle_positions)):
            shift = cycle_positions[i]['phase'] - cycle_positions[i-1]['phase']
            # Normalize to [-π, π]
            phase_shifts.append(np.arctan2(np.sin(shift), np.cos(shift)))
        
        # Determine which base position this cycle corresponds to in the meta-pattern
        meta_position = self._determine_meta_position(cycle, recursion_order)
        
        # Get original position data for comparison
        original_position_data = self._get_meta_position_data(
            position_history, meta_position, recursion_order-1
        )
        
        # Detect resonance with original position
        position_resonance = self._detect_position_resonance(
            cycle_positions, original_position_data
        )
        
        # Analyze energy evolution pattern
        energy_values = [p['energy_level'] for p in cycle_positions]
        energy_pattern = self._analyze_energy_evolution(energy_values)
        
        # Check for phi resonance in energy ratios
        phi_resonance = self._detect_phi_resonance(energy_values)
        
        # Calculate cyclic coherence
        cyclic_coherence = self._calculate_cyclic_coherence(phase_shifts)
        
        # Check for characteristic meta-pattern emergence signature
        meta_signature = self._detect_meta_signature(
            phase_shifts, position_resonance, energy_pattern, phi_resonance, cyclic_coherence
        )
        
        result = {
            'detected': meta_signature['detected'],
            'confidence': meta_signature['confidence'],
            'cycle': cycle,
            'recursion_order': recursion_order,
            'meta_position': meta_position,
            'phase_shift_pattern': phase_shifts,
            'position_resonance': position_resonance,
            'energy_pattern': energy_pattern,
            'phi_resonance': phi_resonance,
            'cyclic_coherence': cyclic_coherence,
            'meta_signature': meta_signature,
            'tick_range': (cycle_positions[0]['tick'], cycle_positions[-1]['tick'])
        }
        
        if result['detected']:
            logger.info(f"Meta-transition detected at cycle {cycle} (order {recursion_order}, " +
                       f"meta-position {meta_position}) with confidence {result['confidence']:.4f}")
            logger.info(f"Ticks: {result['tick_range'][0]} to {result['tick_range'][1]}")
        
        return result
    
    def predict_metacycle_evolution(self, current_order: int, max_prediction_order: int = 5) -> Dict:
        """
        Predict the evolution of meta-patterns into higher orders.
        
        Args:
            current_order: Current meta-pattern order observed
            max_prediction_order: Maximum order to predict
            
        Returns:
            Prediction dictionary
        """
        predictions = {}
        
        for order in range(current_order + 1, max_prediction_order + 1):
            meta3_cycle = self._calculate_meta_position_cycle(3, order)
            meta6_cycle = self._calculate_meta_position_cycle(6, order)
            meta9_cycle = self._calculate_meta_position_cycle(9, order)
            
            predictions[order] = {
                'meta3_cycle': meta3_cycle,
                'meta6_cycle': meta6_cycle,
                'meta9_cycle': meta9_cycle,
                'confidence': 0.8 / (order - current_order + 1)  # Confidence decreases with distance
            }
        
        # Calculate overall prediction confidence
        overall_confidence = sum(p['confidence'] for p in predictions.values()) / len(predictions)
        
        return {
            'current_order': current_order,
            'max_prediction_order': max_prediction_order,
            'predictions': predictions,
            'overall_confidence': overall_confidence
        }
    
    def get_detected_pattern_summary(self) -> Dict:
        """
        Get a summary of all detected patterns at different recursion orders.
        
        Returns:
            Summary dictionary with metrics for all recursion orders
        """
        summary = {
            'highest_detected_order': 0,
            'total_patterns_detected': 0,
            'orders_with_patterns': [],
            'order_strengths': {},
            'propagation_delays': {}
        }
        
        # Count total patterns and find highest order
        total_patterns = 0
        highest_order = 0
        orders_with_patterns = []
        
        for order, data in self.detected_meta_patterns.items():
            if data['patterns_detected'] > 0:
                total_patterns += data['patterns_detected']
                orders_with_patterns.append(order)
                if order > highest_order:
                    highest_order = order
        
        summary['highest_detected_order'] = highest_order
        summary['total_patterns_detected'] = total_patterns
        summary['orders_with_patterns'] = sorted(orders_with_patterns)
        
        # Calculate average strength by order
        for order, data in self.detected_meta_patterns.items():
            if data['cross_scale_correlations']:
                strengths = []
                for corr_data in data['cross_scale_correlations'].values():
                    avg_corr = (corr_data['meta3'] + corr_data['meta6'] + corr_data['meta9']) / 3
                    strengths.append(avg_corr)
                
                if strengths:
                    summary['order_strengths'][order] = np.mean(strengths)
        
        # Include propagation delays
        if self.pattern_propagation:
            summary['propagation_delays'] = {
                str(order_pair): np.mean(delays) 
                for order_pair, delays in self.pattern_propagation.items()
            }
        
        return summary
    
    def _calculate_meta_position_cycle(self, position: int, order: int) -> int:
        """
        Calculate which cycle corresponds to a meta-position at a given order.
        
        Args:
            position: Base position (3, 6, or 9)
            order: Recursion order (1=base level, 2=meta, 3=meta-meta, etc.)
            
        Returns:
            Cycle number
        """
        if order < 1:
            logger.warning(f"Invalid recursion order: {order}, using 1 instead")
            order = 1
            
        if position not in [3, 6, 9]:
            logger.warning(f"Invalid position: {position}, must be 3, 6, or 9")
            return 0
            
        # The formula is: Meta₍ₙ₎(position) = Cycle(position × 2ⁿ⁻¹)
        return position * (2 ** (order - 1))
    
    def _determine_meta_position(self, cycle: int, order: int) -> int:
        """
        Determine which meta-position a cycle represents at a given order.
        
        Args:
            cycle: Cycle number
            order: Recursion order
            
        Returns:
            Base position (3, 6, or 9) or 0 if not a meta-position
        """
        for position in [3, 6, 9]:
            if cycle == self._calculate_meta_position_cycle(position, order):
                return position
                
        return 0  # Not a meta-position at this order
    
    def _get_meta_position_data(self, 
                               position_history: Dict[int, List[Dict]], 
                               position: int, 
                               order: int) -> List[Dict]:
        """
        Get data for a meta-position at a specified order.
        
        Args:
            position_history: Dictionary of position history by recursion depth
            position: Position number (3, 6, or 9)
            order: Recursion order (1=base level, 2=meta, etc.)
            
        Returns:
            List of position data dictionaries
        """
        if order <= 0:
            # Order 0 doesn't exist, interpret as requesting the original position
            return self._extract_position_data(position_history, position)
            
        if order == 1:
            # Order 1 is the base level
            return self._extract_position_data(position_history, position)
            
        # For higher orders, get the corresponding cycle
        cycle = self._calculate_meta_position_cycle(position, order)
        return self._extract_cycle_positions(position_history, cycle)
    
    def _analyze_base_pattern(self, position_history: Dict[int, List[Dict]]) -> Dict:
        """
        Analyze the base level 3-6-9 pattern (recursion order 1).
        
        Args:
            position_history: Dictionary of position history by recursion depth
            
        Returns:
            Dictionary with base pattern metrics
        """
        # Extract position data for positions 3, 6, and 9
        pos3_data = self._extract_position_data(position_history, 3)
        pos6_data = self._extract_position_data(position_history, 6)
        pos9_data = self._extract_position_data(position_history, 9)
        
        # Calculate phase coherence for each position
        pos3_coherence = self._calculate_phase_coherence(pos3_data)
        pos6_coherence = self._calculate_phase_coherence(pos6_data)
        pos9_coherence = self._calculate_phase_coherence(pos9_data)
        
        # Calculate correlations between positions
        pos3_pos6_correlation = self._calculate_meta_correlation(pos3_data, pos6_data)
        pos6_pos9_correlation = self._calculate_meta_correlation(pos6_data, pos9_data)
        pos3_pos9_correlation = self._calculate_meta_correlation(pos3_data, pos9_data)
        
        # Calculate overall pattern strength
        pattern_strength = (pos3_coherence + pos6_coherence + pos9_coherence) / 3
        correlation_strength = (pos3_pos6_correlation + pos6_pos9_correlation + pos3_pos9_correlation) / 3
        overall_strength = 0.6 * pattern_strength + 0.4 * correlation_strength
        
        # Check if base pattern is detected
        is_detected = overall_strength > self.correlation_threshold
        
        return {
            'detected': is_detected,
            'recursion_order': 1,
            'pattern_strength': overall_strength,
            'pos3_coherence': pos3_coherence,
            'pos6_coherence': pos6_coherence,
            'pos9_coherence': pos9_coherence,
            'pos3_pos6_correlation': pos3_pos6_correlation,
            'pos6_pos9_correlation': pos6_pos9_correlation,
            'pos3_pos9_correlation': pos3_pos9_correlation,
            'pos3_data': {
                'count': len(pos3_data),
                'energy_pattern': self._extract_energy_pattern(pos3_data)
            },
            'pos6_data': {
                'count': len(pos6_data),
                'energy_pattern': self._extract_energy_pattern(pos6_data)
            },
            'pos9_data': {
                'count': len(pos9_data),
                'energy_pattern': self._extract_energy_pattern(pos9_data)
            }
        }
    
    def _analyze_pattern_propagation(self, all_results: Dict[int, Dict]) -> None:
        """
        Analyze how patterns propagate between recursion levels.
        
        Args:
            all_results: Dictionary of detection results by recursion order
        """
        # Reset propagation data
        self.pattern_propagation = defaultdict(list)
        
        # Skip if we don't have at least two levels of patterns
        detected_orders = [order for order, result in all_results.items() if result.get('detected', False)]
        if len(detected_orders) < 2:
            return
            
        # Sort orders
        detected_orders.sort()
        
        # Calculate delays between consecutive orders
        for i in range(len(detected_orders) - 1):
            lower_order = detected_orders[i]
            higher_order = detected_orders[i + 1]
            
            # Get the tick ranges for pattern detection
            lower_tick_range = self._get_earliest_detection_tick(all_results[lower_order])
            higher_tick_range = self._get_earliest_detection_tick(all_results[higher_order])
            
            if lower_tick_range and higher_tick_range:
                # Calculate propagation delay
                propagation_delay = higher_tick_range - lower_tick_range
                
                # Store the propagation delay
                self.pattern_propagation[(lower_order, higher_order)].append(propagation_delay)
                
                logger.info(f"Pattern propagation from order {lower_order} to {higher_order} " +
                           f"took {propagation_delay} ticks")
    
    def _get_earliest_detection_tick(self, result: Dict) -> Optional[int]:
        """
        Get the earliest tick when a pattern was detected.
        
        Args:
            result: Pattern detection result dictionary
            
        Returns:
            Earliest tick or None if not available
        """
        # For recursion order 1 (base level)
        if result.get('recursion_order') == 1:
            # Use the earliest tick from positions 3, 6, 9
            ticks = []
            for pos in ['pos3', 'pos6', 'pos9']:
                data = result.get(f'{pos}_data', {})
                if 'energy_pattern' in data and data['energy_pattern']:
                    ticks.append(min(range(len(data['energy_pattern']))))
            
            return min(ticks) if ticks else None
        
        # For higher recursion orders
        for meta_pos in ['meta3', 'meta6', 'meta9']:
            data = result.get(f'{meta_pos}_data', {})
            if data and 'energy_pattern' in data and data['energy_pattern']:
                if 'tick_range' in result:
                    return result['tick_range'][0]  # Start of tick range
                    
        return None
    
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