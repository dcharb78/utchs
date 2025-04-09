"""
Coherence gating module for the UTCHS framework.

This module implements the CoherenceGate class, which filters patterns based on 
coherence metrics to prevent noise amplification at higher recursion orders.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class CoherenceGate:
    """
    Implements coherence gating to filter stable patterns.
    
    This class provides mechanisms to filter detected patterns based on coherence
    metrics, preventing noise amplification at higher recursion orders and
    ensuring only stable patterns propagate through the detection system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the coherence gate.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        
        # Coherence thresholds
        self.base_threshold = self.config.get('base_threshold', 0.7)
        self.recursion_factor = self.config.get('recursion_factor', 0.1)
        self.min_threshold = self.config.get('min_threshold', 0.4)
        
        # Importance weights for different coherence metrics
        self.phase_weight = self.config.get('phase_weight', 0.5)
        self.energy_weight = self.config.get('energy_weight', 0.3)
        self.temporal_weight = self.config.get('temporal_weight', 0.2)
        
        # Storage for coherence history
        self.coherence_history = defaultdict(list)
        
        # Configure maximum history length
        self.max_history_length = self.config.get('max_history_length', 1000)
        
        # Adaptive threshold parameters
        self.enable_adaptive_threshold = self.config.get('enable_adaptive_threshold', True)
        self.adaptation_rate = self.config.get('adaptation_rate', 0.1)
        
        # Store adaptive thresholds by recursion order
        self.adaptive_thresholds = {}
        
        logger.info(f"CoherenceGate initialized with base threshold: {self.base_threshold:.2f}")
    
    def calculate_threshold(self, recursion_order: int) -> float:
        """
        Calculate adaptive coherence threshold based on recursion order.
        
        Args:
            recursion_order: Current recursion order
            
        Returns:
            Coherence threshold for this recursion order
        """
        # Check if we have an adaptive threshold for this order
        if self.enable_adaptive_threshold and recursion_order in self.adaptive_thresholds:
            return self.adaptive_thresholds[recursion_order]
        
        # Higher recursion orders have lower threshold requirements
        # This is because higher-order patterns naturally have more variability
        threshold = self.base_threshold - (recursion_order - 1) * self.recursion_factor
        threshold = max(threshold, self.min_threshold)
        
        logger.debug(f"Coherence threshold for recursion order {recursion_order}: {threshold:.4f}")
        
        return threshold
    
    def is_coherent(self, pattern_data: Dict[str, Any], recursion_order: int) -> bool:
        """
        Determine if a pattern passes the coherence gate.
        
        Args:
            pattern_data: Pattern metrics
            recursion_order: Current recursion order
            
        Returns:
            Boolean indicating if pattern passes gate
        """
        threshold = self.calculate_threshold(recursion_order)
        
        # Calculate combined coherence score using weighted components
        phase_coherence = pattern_data.get('phase_coherence', 0)
        energy_coherence = pattern_data.get('energy_coherence', 0)
        temporal_coherence = pattern_data.get('temporal_coherence', 0)
        
        # Apply weights to coherence components
        coherence_score = (
            self.phase_weight * phase_coherence +
            self.energy_weight * energy_coherence +
            self.temporal_weight * temporal_coherence
        )
        
        # Store for historical analysis
        self.coherence_history[recursion_order].append(coherence_score)
        
        # Limit history length
        if len(self.coherence_history[recursion_order]) > self.max_history_length:
            self.coherence_history[recursion_order].pop(0)
        
        # Adaptive threshold update if enabled
        if self.enable_adaptive_threshold and len(self.coherence_history[recursion_order]) > 10:
            self._update_adaptive_threshold(recursion_order)
        
        # Log coherence results for debugging
        is_coherent = coherence_score >= threshold
        if not is_coherent:
            logger.debug(f"Pattern rejected at order {recursion_order} - " +
                       f"Score: {coherence_score:.4f}, Threshold: {threshold:.4f}")
        
        return is_coherent
    
    def filter_patterns(self, patterns: List[Dict], recursion_order: int) -> List[Dict]:
        """
        Filter patterns by coherence threshold.
        
        Args:
            patterns: List of detected patterns
            recursion_order: Current recursion order
            
        Returns:
            Filtered list of patterns that pass gate
        """
        coherent_patterns = [p for p in patterns if self.is_coherent(p, recursion_order)]
        
        logger.info(f"Filtered {len(patterns) - len(coherent_patterns)} of {len(patterns)} " +
                  f"patterns at recursion order {recursion_order}")
        
        return coherent_patterns
    
    def get_coherence_metrics(self, recursion_order: Optional[int] = None) -> Dict:
        """
        Get coherence metrics for analysis.
        
        Args:
            recursion_order: Optional specific recursion order to get metrics for
                             If None, returns metrics for all orders
                             
        Returns:
            Dictionary with coherence metrics
        """
        if recursion_order is not None:
            if recursion_order not in self.coherence_history:
                return {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0,
                    'threshold': self.calculate_threshold(recursion_order)
                }
            
            history = self.coherence_history[recursion_order]
            return {
                'mean': float(np.mean(history)) if history else 0.0,
                'std': float(np.std(history)) if history else 0.0,
                'min': float(np.min(history)) if history else 0.0,
                'max': float(np.max(history)) if history else 0.0,
                'count': len(history),
                'threshold': self.calculate_threshold(recursion_order)
            }
        
        # Get metrics for all recursion orders
        all_metrics = {}
        for order in sorted(self.coherence_history.keys()):
            all_metrics[order] = self.get_coherence_metrics(order)
        
        return all_metrics
    
    def _update_adaptive_threshold(self, recursion_order: int) -> None:
        """
        Update adaptive threshold based on coherence history.
        
        Args:
            recursion_order: Recursion order to update threshold for
        """
        if recursion_order not in self.coherence_history:
            return
        
        history = self.coherence_history[recursion_order]
        if not history:
            return
        
        # Calculate statistics from history
        mean_coherence = np.mean(history)
        std_coherence = np.std(history)
        
        # Set threshold based on historical patterns
        # Using mean - 1*std ensures we capture ~84% of patterns that match historical coherence
        new_threshold = mean_coherence - std_coherence
        
        # Ensure threshold stays within allowed range
        base_threshold = self.base_threshold - (recursion_order - 1) * self.recursion_factor
        new_threshold = max(new_threshold, self.min_threshold)
        new_threshold = min(new_threshold, base_threshold)
        
        # If we already have an adaptive threshold, blend with existing
        if recursion_order in self.adaptive_thresholds:
            current = self.adaptive_thresholds[recursion_order]
            new_threshold = (1 - self.adaptation_rate) * current + self.adaptation_rate * new_threshold
        
        # Update the adaptive threshold
        self.adaptive_thresholds[recursion_order] = new_threshold
        
        logger.debug(f"Updated adaptive threshold for order {recursion_order}: {new_threshold:.4f}")
    
    def reset_history(self) -> None:
        """Reset coherence history and adaptive thresholds."""
        self.coherence_history.clear()
        self.adaptive_thresholds.clear()
        
        logger.info("CoherenceGate history and adaptive thresholds reset")
    
    def calculate_combined_coherence(self, 
                                    phase_coherence: float, 
                                    energy_coherence: float, 
                                    temporal_coherence: float) -> float:
        """
        Calculate combined coherence score from individual metrics.
        
        Args:
            phase_coherence: Phase coherence value (0.0-1.0)
            energy_coherence: Energy pattern coherence value (0.0-1.0)
            temporal_coherence: Temporal coherence value (0.0-1.0)
            
        Returns:
            Combined coherence score (0.0-1.0)
        """
        # Apply weights to coherence components
        coherence_score = (
            self.phase_weight * phase_coherence +
            self.energy_weight * energy_coherence +
            self.temporal_weight * temporal_coherence
        )
        
        return coherence_score

# Create a global instance for easy import
coherence_gate = CoherenceGate()

def is_pattern_coherent(pattern_data: Dict[str, Any], recursion_order: int) -> bool:
    """
    Global function to check if a pattern passes the coherence gate.
    
    Args:
        pattern_data: Pattern metrics
        recursion_order: Current recursion order
        
    Returns:
        Boolean indicating if pattern passes gate
    """
    return coherence_gate.is_coherent(pattern_data, recursion_order) 