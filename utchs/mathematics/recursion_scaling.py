"""
Recursion scaling module for the UTCHS framework.

This module implements nonlinear correction functions to improve the accuracy
of recursive meta-pattern detection at higher recursion orders.
"""

import math
import numpy as np
from typing import Dict, Optional, Tuple, Any, Callable, Union

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class RecursionScaling:
    """
    Implements nonlinear scaling functions for recursion orders in the UTCHS framework.
    
    This class provides correction factors to account for nonlinear effects
    that emerge at higher recursion orders, ensuring more accurate detection
    of meta-patterns within the infinite pattern detection framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the recursion scaling.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        
        # Configuration parameters
        self.base_correction = self.config.get('base_correction', True)
        self.logarithmic_dampening = self.config.get('logarithmic_dampening', True)
        self.stability_adjustment = self.config.get('stability_adjustment', True)
        
        # Correction scaling parameters
        self.log_dampening_factor = self.config.get('log_dampening_factor', 10.0)
        self.stability_threshold = self.config.get('stability_threshold', 0.7)
        self.max_correction = self.config.get('max_correction', 0.5)
        
        # Map of scaling function names to implementations
        self.scaling_functions = {
            'linear': self.linear_scaling,
            'logarithmic': self.logarithmic_scaling,
            'adaptive': self.adaptive_scaling,
            'phi_resonance': self.phi_resonance_scaling
        }
        
        # Default scaling function
        self.default_scaling = self.config.get('default_scaling', 'logarithmic')
        
        # Numeric stability safeguards
        self.min_correction = self.config.get('min_correction', 0.01)
        
        logger.info(f"RecursionScaling initialized with default scaling: {self.default_scaling}")
        
    def get_correction_factor(self, 
                              recursion_order: int, 
                              system_state: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate the nonlinear correction factor for a given recursion order.
        
        Args:
            recursion_order: Current recursion order (n)
            system_state: Optional system state for adaptive corrections
            
        Returns:
            Correction factor to apply to meta-cycle calculation
        """
        if recursion_order <= 2:
            # No correction needed for lower orders
            return 1.0
        
        # Default correction using logarithmic dampening
        correction_factor = 1.0
        
        # Apply appropriate scaling function
        scaling_function = self.scaling_functions.get(
            self.default_scaling, self.logarithmic_scaling
        )
        
        # Calculate correction
        correction_factor = scaling_function(recursion_order, system_state)
        
        # Apply stability adjustment if enabled and system state is provided
        if self.stability_adjustment and system_state:
            stability_factor = self._calculate_stability_factor(system_state, recursion_order)
            correction_factor *= stability_factor
        
        # Ensure correction stays within reasonable bounds
        correction_factor = max(1.0 - self.max_correction, min(correction_factor, 1.0))
        
        logger.debug(f"Recursion order {recursion_order} correction factor: {correction_factor:.4f}")
        
        return correction_factor
    
    def linear_scaling(self, 
                       recursion_order: int, 
                       system_state: Optional[Dict[str, Any]] = None) -> float:
        """
        Linear scaling function (mainly for testing/comparison).
        
        Args:
            recursion_order: Current recursion order (n)
            system_state: Optional system state (unused in linear scaling)
            
        Returns:
            Correction factor
        """
        # Simple linear decay with order
        return 1.0 - (recursion_order - 2) * 0.05
    
    def logarithmic_scaling(self, 
                            recursion_order: int, 
                            system_state: Optional[Dict[str, Any]] = None) -> float:
        """
        Logarithmic scaling function for dampening at higher orders.
        
        Args:
            recursion_order: Current recursion order (n)
            system_state: Optional system state (unused in logarithmic scaling)
            
        Returns:
            Correction factor using logarithmic dampening
        """
        # Logarithmic dampening at higher orders
        # Formula: 1 - (log(order) / (dampening_factor * order))
        if not self.logarithmic_dampening:
            return 1.0
            
        # Apply logarithmic correction with protection against very small values
        correction = 1.0 - (math.log(max(recursion_order, 2)) / 
                          (self.log_dampening_factor * recursion_order))
                          
        # Ensure minimum correction value for numerical stability
        return max(correction, 1.0 - self.max_correction)
    
    def adaptive_scaling(self, 
                         recursion_order: int, 
                         system_state: Optional[Dict[str, Any]] = None) -> float:
        """
        Adaptive scaling based on current system state and recursion order.
        
        Args:
            recursion_order: Current recursion order (n)
            system_state: System state for adaptive calculation
            
        Returns:
            Correction factor adapted to current system state
        """
        # Start with logarithmic correction
        base_correction = self.logarithmic_scaling(recursion_order, system_state)
        
        # If no system state provided, fall back to base correction
        if not system_state:
            return base_correction
            
        # Extract relevant metrics from system state
        coherence = system_state.get('global_coherence', 1.0)
        stability = system_state.get('global_stability', 1.0)
        energy_level = system_state.get('energy_level', 1.0)
        
        # Calculate adaptive component based on system metrics
        coherence_weight = 0.4
        stability_weight = 0.4
        energy_weight = 0.2
        
        # Metrics closer to 1.0 indicate a more stable system requiring less correction
        adaptive_component = (
            coherence_weight * coherence +
            stability_weight * stability +
            energy_weight * min(energy_level, 1.0)
        )
        
        # Scale adaptive component to correction range
        adaptive_correction = 1.0 - (1.0 - base_correction) * (1.0 - adaptive_component)
        
        return adaptive_correction
    
    def phi_resonance_scaling(self, 
                              recursion_order: int, 
                              system_state: Optional[Dict[str, Any]] = None) -> float:
        """
        Phi-based resonance scaling for harmonic alignment with the golden ratio.
        
        Args:
            recursion_order: Current recursion order (n)
            system_state: Optional system state for phi resonance calculation
            
        Returns:
            Correction factor based on phi-resonance
        """
        # Golden ratio (φ)
        phi = (1 + math.sqrt(5)) / 2
        
        # Calculate how close the recursion order is to a Fibonacci sequence
        # Approximate formula: φⁿ ≈ Fib(n+1)
        power = recursion_order - 1
        fib_approx = phi ** power
        
        # Find closest Fibonacci number
        closest_fib = round(fib_approx)
        
        # Calculate resonance (how close we are to a "perfect" Fibonacci recursion)
        resonance = 1.0 - min(abs(fib_approx - closest_fib) / fib_approx, 0.5)
        
        # Apply resonance to correction (more resonant = less correction needed)
        base_correction = self.logarithmic_scaling(recursion_order, system_state)
        resonant_correction = base_correction * (0.8 + 0.2 * resonance)
        
        return resonant_correction
    
    def _calculate_stability_factor(self, 
                                   system_state: Dict[str, Any], 
                                   recursion_order: int) -> float:
        """
        Calculate stability factor based on system state.
        
        Args:
            system_state: Current system state
            recursion_order: Current recursion order
            
        Returns:
            Stability factor (0.0-1.0)
        """
        # Extract stability metrics
        coherence = system_state.get('global_coherence', 0.0)
        energy_stability = system_state.get('energy_stability', 0.0)
        phase_stability = system_state.get('phase_stability', 0.0)
        
        # Calculate combined stability
        combined_stability = (
            0.5 * coherence +
            0.3 * energy_stability +
            0.2 * phase_stability
        )
        
        # More stable systems require less correction
        if combined_stability >= self.stability_threshold:
            return 1.0
        
        # Scale stability factor based on distance from threshold
        stability_factor = 0.8 + 0.2 * (combined_stability / self.stability_threshold)
        
        return stability_factor

# Create a global instance for easy import
recursion_scaling = RecursionScaling()

def get_correction_factor(recursion_order: int, 
                          system_state: Optional[Dict[str, Any]] = None) -> float:
    """
    Global function to get nonlinear correction factor for a recursion order.
    
    Args:
        recursion_order: Current recursion order (n)
        system_state: Optional system state for adaptive corrections
        
    Returns:
        Correction factor
    """
    return recursion_scaling.get_correction_factor(recursion_order, system_state) 