"""
UTCHS Framework - Enhanced Unified Toroidal-Crystalline Harmonic System

A comprehensive theoretical model that integrates concepts of toroidal geometry,
crystalline structures, harmonic systems, and phase recursion.
"""

__version__ = "0.1.0"

from .core.system import UTCHSSystem
from .core.torus import Torus
from .math.phase_field import PhaseField
from .fields.energy_field import EnergyField

__all__ = [
    "UTCHSSystem",
    "Torus",
    "PhaseField",
    "EnergyField",
]
