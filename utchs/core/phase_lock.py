"""
Phase locking module for the UTCHS framework.

This module implements the TorsionalPhaseLock class, which provides phase-locking 
mechanisms between recursion levels to prevent destructive interference.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class TorsionalPhaseLock:
    """
    Implements torsional phase-locking between recursion levels.
    
    This class aligns phases between different recursion levels to prevent
    destructive interference between 13D systems that emerge at different
    recursion orders. The phase-locking mechanism ensures coherent interaction
    between systems by applying torsional adjustments to phase values.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the torsional phase-locking system.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.phase_tolerance = self.config.get('phase_tolerance', 0.1)
        self.lock_strength = self.config.get('lock_strength', 0.7)
        
        # Storage for phase history and adjustments
        self.phase_history = defaultdict(list)
        self.adjustment_history = defaultdict(list)
        
        # Tracking for phase coherence metrics
        self.coherence_metrics = defaultdict(lambda: defaultdict(float))
        
        # Configure maximum history length
        self.max_history_length = self.config.get('max_history_length', 1000)
        
        # Numeric stability parameters
        self.min_phase_magnitude = self.config.get('min_phase_magnitude', 1e-10)
        
        logger.info("TorsionalPhaseLock initialized with lock_strength: " +
                  f"{self.lock_strength:.2f}")
    
    def align_phases(self, base_phase: complex, target_phase: complex) -> complex:
        """
        Align target phase with base phase using torsional adjustment.
        
        Args:
            base_phase: Reference phase to align to
            target_phase: Phase to be aligned
            
        Returns:
            Aligned phase value
        """
        # Handle zero or near-zero phases with numeric stability
        if abs(base_phase) < self.min_phase_magnitude or abs(target_phase) < self.min_phase_magnitude:
            return target_phase
        
        # Calculate phase difference
        # Using np.angle() to handle complex numbers correctly
        phase_diff = np.angle(target_phase / base_phase)
        
        # Apply torsional correction with lock strength as scaling factor
        # Only partially adjust to avoid over-correction
        correction = self.lock_strength * phase_diff
        
        # Apply correction through complex rotation
        aligned_phase = target_phase * np.exp(-1j * correction)
        
        # Log significant adjustments
        if abs(correction) > 0.5:  # Only log substantial corrections
            logger.debug(f"Applied phase correction of {correction:.4f} radians " +
                       f"(strength: {self.lock_strength:.2f})")
        
        return aligned_phase
    
    def align_recursion_levels(self, 
                              position_data: Dict[int, Dict[int, Dict]], 
                              base_level: int, 
                              target_level: int) -> Dict[int, Dict[int, Dict]]:
        """
        Align phases between recursion levels.
        
        Args:
            position_data: Dictionary of position data by recursion level
            base_level: Reference recursion level
            target_level: Recursion level to align
            
        Returns:
            Aligned position data for target level
        """
        if base_level not in position_data or target_level not in position_data:
            return position_data
        
        # Track alignments and their coherence
        positions_aligned = 0
        total_coherence = 0.0
        
        # Get base positions (3, 6, 9) - these are fundamental triplet positions
        for position in [3, 6, 9]:
            if (position in position_data[base_level] and 
                position in position_data[target_level]):
                
                base_phase = position_data[base_level][position].get('phase')
                target_phase = position_data[target_level][position].get('phase')
                
                if base_phase is not None and target_phase is not None:
                    # Align phases
                    aligned_phase = self.align_phases(base_phase, target_phase)
                    position_data[target_level][position]['phase'] = aligned_phase
                    
                    # Store phase adjustment for history
                    adjustment = np.angle(aligned_phase / target_phase)
                    self.adjustment_history[target_level].append(adjustment)
                    
                    # Limit history length
                    if len(self.adjustment_history[target_level]) > self.max_history_length:
                        self.adjustment_history[target_level].pop(0)
                    
                    # Calculate phase coherence after alignment
                    coherence = self._calculate_phase_coherence(base_phase, aligned_phase)
                    total_coherence += coherence
                    positions_aligned += 1
        
        # Update coherence metrics
        if positions_aligned > 0:
            avg_coherence = total_coherence / positions_aligned
            self.coherence_metrics[base_level][target_level] = avg_coherence
            
            logger.debug(f"Aligned recursion levels {base_level} → {target_level} " +
                       f"with coherence: {avg_coherence:.4f}")
        
        return position_data
    
    def align_multi_level_phases(self, 
                                position_data: Dict[int, Dict[int, Dict]]) -> Dict[int, Dict[int, Dict]]:
        """
        Align phases across multiple recursion levels.
        
        This method aligns all recursion levels starting from the lowest (base) level
        and propagating adjustments upward through the hierarchy.
        
        Args:
            position_data: Dictionary of position data by recursion level
            
        Returns:
            Aligned position data across all levels
        """
        # Get all recursion levels in ascending order
        recursion_levels = sorted(position_data.keys())
        
        if len(recursion_levels) <= 1:
            return position_data  # Nothing to align with only one level
        
        # Start with the lowest level as the base
        base_level = recursion_levels[0]
        
        # Align each higher level with the level below it
        for i in range(1, len(recursion_levels)):
            target_level = recursion_levels[i]
            position_data = self.align_recursion_levels(
                position_data, base_level, target_level
            )
            base_level = target_level  # The aligned level becomes the new base
        
        return position_data
    
    def get_phase_coherence_metrics(self) -> Dict[int, Dict[int, float]]:
        """
        Get phase coherence metrics between recursion levels.
        
        Returns:
            Dictionary of coherence metrics indexed by recursion levels
        """
        return dict(self.coherence_metrics)
    
    def _calculate_phase_coherence(self, phase1: complex, phase2: complex) -> float:
        """
        Calculate phase coherence between two complex phase values.
        
        Args:
            phase1, phase2: Complex phase values
            
        Returns:
            Coherence value (0.0-1.0)
        """
        # Handle zero or near-zero phases
        if abs(phase1) < self.min_phase_magnitude or abs(phase2) < self.min_phase_magnitude:
            return 0.0
        
        # Convert to unit vectors on complex plane
        v1 = phase1 / abs(phase1)
        v2 = phase2 / abs(phase2)
        
        # Calculate coherence as normalized dot product
        # This gives 1.0 for perfect alignment and 0.0 for orthogonal phases
        coherence = abs(v1.real * v2.real + v1.imag * v2.imag)
        
        # Normalize to [0, 1]
        coherence = min(max(coherence, 0.0), 1.0)
        
        return coherence
    
    def analyze_adjustment_stability(self, recursion_level: int) -> Dict[str, float]:
        """
        Analyze the stability of phase adjustments for a recursion level.
        
        Args:
            recursion_level: Recursion level to analyze
            
        Returns:
            Dictionary with stability metrics
        """
        if recursion_level not in self.adjustment_history or not self.adjustment_history[recursion_level]:
            return {
                'stability': 1.0,
                'mean_adjustment': 0.0,
                'adjustment_std': 0.0,
                'adjustment_count': 0
            }
        
        adjustments = np.array(self.adjustment_history[recursion_level])
        
        # Calculate statistics
        mean_adjustment = np.mean(adjustments)
        std_adjustment = np.std(adjustments)
        
        # Stability is inversely proportional to adjustment variability
        # More variable adjustments indicate less stable phase-locking
        stability = 1.0 / (1.0 + std_adjustment)
        
        return {
            'stability': float(stability),
            'mean_adjustment': float(mean_adjustment),
            'adjustment_std': float(std_adjustment),
            'adjustment_count': len(adjustments)
        }
    
    def reset_history(self) -> None:
        """Reset all history and metrics."""
        self.phase_history.clear()
        self.adjustment_history.clear()
        self.coherence_metrics.clear()
        
        logger.info("TorsionalPhaseLock history and metrics reset")

# Create a global instance for easy import
torsional_phase_lock = TorsionalPhaseLock()

def align_phases(base_phase: complex, target_phase: complex) -> complex:
    """
    Global function to align phases using torsional adjustment.
    
    Args:
        base_phase: Reference phase to align to
        target_phase: Phase to be aligned
        
    Returns:
        Aligned phase value
    """
    return torsional_phase_lock.align_phases(base_phase, target_phase) 