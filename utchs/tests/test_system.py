"""
Test module for the UTCHS system.

This module contains basic tests to verify system initialization and basic functionality.
"""

import pytest
import numpy as np
from typing import Dict, Any

from ..core.system import UTCHSSystem

def test_system_initialization():
    """Test that the system initializes correctly with default configuration."""
    config: Dict[str, Any] = {
        "grid_size": (10, 10, 10),
        "grid_spacing": 0.1,
        "history_length": 5,
        "initial_pattern": "random"
    }
    
    system = UTCHSSystem(config)
    
    assert system.current_tick == 0
    assert len(system.tori) == 1
    assert system.current_torus_idx == 0
    assert system.global_coherence == 1.0
    assert system.global_stability == 0.0
    assert system.energy_level == 0.0
    assert system.phase_recursion_depth == 1

def test_phase_field_initialization():
    """Test that the phase field initializes correctly."""
    config: Dict[str, Any] = {
        "grid_size": (10, 10, 10),
        "grid_spacing": 0.1,
        "history_length": 5,
        "initial_pattern": "random"
    }
    
    system = UTCHSSystem(config)
    
    assert system.phase_field.grid_size == (10, 10, 10)
    assert system.phase_field.dx == 0.1
    assert system.phase_field.field.shape == (10, 10, 10)
    assert system.phase_field.phi_components.shape == (3, 10, 10, 10) 