"""
Recursion tracker module for the UTCHS framework.

This module implements the RecursionTracker class, which tracks position history
across multiple recursive levels (octaves).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import deque

from ..core.position import Position
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class RecursionTracker:
    """
    Tracks position history across multiple recursive levels (octaves).
    
    This class maintains a record of positions as they transition through
    different recursive levels, allowing for analysis of patterns that 
    span across octaves.
    
    This class implements the Singleton pattern to ensure all components
    access the same tracking instance.
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None) -> 'RecursionTracker':
        """
        Get the singleton instance of RecursionTracker.
        
        Args:
            config: Configuration dictionary (optional), only used when creating 
                   the instance for the first time
                   
        Returns:
            Singleton RecursionTracker instance
        """
        if cls._instance is None:
            cls._instance = cls(config)
            logger.info("Created RecursionTracker singleton instance")
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the recursion tracker.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.max_history_length = self.config.get('recursion_history_length', 1000)
        self.max_recursion_depth = self.config.get('max_recursion_depth', 7)
        
        # Position history by recursion level
        self.position_history = {depth: deque(maxlen=self.max_history_length) 
                                for depth in range(self.max_recursion_depth + 1)}
        
        # Track positions by their absolute identifier
        self.absolute_position_history = {}
        
        # Track transitions between recursion levels
        self.recursion_transitions = []
        
        # Golden ratio (φ) for scaling calculations
        self.phi = (1 + np.sqrt(5)) / 2
        
        logger.info(f"RecursionTracker initialized with max_recursion_depth={self.max_recursion_depth}")
    
    def reset(self):
        """
        Reset the recursion tracker state.
        
        This clears all position history and transitions.
        """
        # Reset position history by recursion level
        self.position_history = {depth: deque(maxlen=self.max_history_length) 
                               for depth in range(self.max_recursion_depth + 1)}
        
        # Reset position tracking by absolute identifier
        self.absolute_position_history = {}
        
        # Reset transitions between recursion levels
        self.recursion_transitions = []
        
        logger.info("RecursionTracker state reset")
    
    def track_position(self, position: Position, recursion_depth: int, tick: int) -> None:
        """
        Track a position at a specific recursion depth.
        
        Args:
            position: Position object to track
            recursion_depth: Current recursion depth
            tick: Current system tick
        """
        if recursion_depth > self.max_recursion_depth:
            logger.warning(f"Recursion depth {recursion_depth} exceeds maximum ({self.max_recursion_depth})")
            return
        
        # Create position snapshot
        position_data = {
            'tick': tick,
            'position_number': position.number,
            'energy_level': position.energy_level,
            'phase': position.phase,
            'absolute_position': position.absolute_position.copy(),
            'spatial_location': position.spatial_location.copy(),
            'recursion_depth': recursion_depth
        }
        
        # Add to history for this recursion depth
        self.position_history[recursion_depth].append(position_data)
        
        # Create a unique identifier for this absolute position
        absolute_id = self._get_absolute_position_id(position.absolute_position)
        
        # Track by absolute position
        if absolute_id not in self.absolute_position_history:
            self.absolute_position_history[absolute_id] = deque(maxlen=self.max_history_length)
        
        self.absolute_position_history[absolute_id].append(position_data)
        
        # Check for position 10 as recursive seed point
        if position.number == 10:
            logger.debug(f"Position 10 detected at depth {recursion_depth}, tick {tick}")
            self._check_for_recursion_transition(position_data)
    
    def _get_absolute_position_id(self, absolute_position: Dict) -> str:
        """
        Generate a unique identifier for an absolute position.
        
        Args:
            absolute_position: Dictionary with torus, structure, cycle, position
            
        Returns:
            String identifier
        """
        return f"T{absolute_position['torus']}_S{absolute_position['structure']}_C{absolute_position['cycle']}_P{absolute_position['position']}"
    
    def _check_for_recursion_transition(self, position_data: Dict) -> None:
        """
        Check if a recursion transition is occurring at position 10.
        
        Args:
            position_data: Current position data
        """
        recursion_depth = position_data['recursion_depth']
        
        # We need previous P10 data to compare
        abs_id = self._get_absolute_position_id(position_data['absolute_position'])
        history = self.absolute_position_history.get(abs_id, None)
        
        if not history or len(history) < 2:
            return
        
        # Get previous P10 data
        previous_data = history[-2]
        
        # Calculate transition metrics
        phase_shift = position_data['phase'] - previous_data['phase']
        energy_ratio = position_data['energy_level'] / previous_data['energy_level'] if previous_data['energy_level'] != 0 else 0
        
        # Check for phi-based scaling (golden ratio resonance)
        phi_phase_resonance = abs(abs(phase_shift) - (1/self.phi)) < 0.1
        phi_energy_resonance = abs(energy_ratio - self.phi) < 0.2
        
        # Record transition
        transition = {
            'tick': position_data['tick'],
            'recursion_depth': recursion_depth,
            'phase_shift': phase_shift,
            'energy_ratio': energy_ratio,
            'phi_phase_resonance': phi_phase_resonance,
            'phi_energy_resonance': phi_energy_resonance,
            'position': position_data.copy(),
            'previous_position': previous_data.copy()
        }
        
        self.recursion_transitions.append(transition)
        
        if phi_phase_resonance or phi_energy_resonance:
            logger.info(f"Golden ratio recursion transition detected at depth {recursion_depth}, tick {position_data['tick']}")
    
    def get_position_history(self, position_number: int = None, recursion_depth: Optional[int] = None) -> List[Dict]:
        """
        Get history for positions, optionally filtered by position number and recursion depth.
        
        Args:
            position_number: Position number (1-13), or None for all positions
            recursion_depth: Recursion depth (or None for all depths)
            
        Returns:
            List of position data dictionaries
        """
        result = []
        
        # If recursion depth specified, only check that level
        if recursion_depth is not None:
            if recursion_depth in self.position_history:
                for pos_data in self.position_history[recursion_depth]:
                    if position_number is None or pos_data['position_number'] == position_number:
                        result.append(pos_data)
            return result
        
        # Otherwise check all recursion depths
        for depth, history in self.position_history.items():
            for pos_data in history:
                if position_number is None or pos_data['position_number'] == position_number:
                    result.append(pos_data)
        
        return result
    
    def get_recursion_transitions(self, min_depth: int = 0, max_depth: Optional[int] = None) -> List[Dict]:
        """
        Get recursion transitions, optionally filtered by depth range.
        
        Args:
            min_depth: Minimum recursion depth
            max_depth: Maximum recursion depth (or None for no maximum)
            
        Returns:
            List of transition dictionaries
        """
        if max_depth is None:
            max_depth = self.max_recursion_depth
            
        return [t for t in self.recursion_transitions 
                if min_depth <= t['recursion_depth'] <= max_depth]
    
    def get_position_across_recursion_levels(self, position_number: int, tick: int) -> Dict[int, Dict]:
        """
        Get a specific position across all recursion levels at a specific tick.
        
        Args:
            position_number: Position number (1-13)
            tick: System tick
            
        Returns:
            Dictionary mapping recursion depth to position data
        """
        result = {}
        
        for depth, history in self.position_history.items():
            # Find position data closest to the given tick
            closest_data = None
            min_tick_diff = float('inf')
            
            for pos_data in history:
                if pos_data['position_number'] == position_number:
                    tick_diff = abs(pos_data['tick'] - tick)
                    if tick_diff < min_tick_diff:
                        min_tick_diff = tick_diff
                        closest_data = pos_data
            
            if closest_data:
                result[depth] = closest_data
        
        return result 