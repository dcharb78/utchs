# UTCHS Nonlinear Recursion Enhancements

This document provides an overview of the nonlinear recursion enhancements implemented in the UTCHS framework, along with usage examples and integration points.

## Overview

The nonlinear recursion enhancements address several limitations in the original linear model:

1. **Nonlinear Recursion Modeling**: Implements nonlinear correction terms for meta-cycle calculations to account for effects that emerge at higher recursion orders.
   
2. **Möbius Correction Terms**: Enhances Möbius transformations with correction parameters that adapt to recursion depth, improving the accuracy of spatial and phase transformations.
   
3. **Torsional Phase-Locking**: Provides phase alignment between recursion levels to prevent destructive interference between 13D systems.
   
4. **Coherence Gating**: Filters patterns based on multi-dimensional coherence metrics, preventing noise amplification at higher recursion orders.

## Components

### 1. RecursionScaling

The `RecursionScaling` class provides nonlinear correction factors for meta-cycle calculations:

```python
from utchs.mathematics.recursion_scaling import RecursionScaling, get_correction_factor

# Using the class directly
rs = RecursionScaling(config={'logarithmic_dampening': True})
correction = rs.get_correction_factor(recursion_order=4)

# Using the global function (simpler)
correction = get_correction_factor(recursion_order=4, system_state=system_state)
```

#### Configuration Options:

- `logarithmic_dampening`: Enable logarithmic dampening (default: True)
- `stability_adjustment`: Enable stability-based adjustment (default: True)
- `log_dampening_factor`: Strength of logarithmic dampening (default: 10.0)
- `default_scaling`: Scaling function to use (default: 'logarithmic')
- `stability_threshold`: Threshold for stability adjustment (default: 0.7)

### 2. TorsionalPhaseLock

The `TorsionalPhaseLock` class provides phase alignment between recursion levels:

```python
from utchs.core.phase_lock import TorsionalPhaseLock, align_phases

# Using the class
tpl = TorsionalPhaseLock(config={'lock_strength': 0.8})
aligned_data = tpl.align_recursion_levels(position_data, base_level=1, target_level=2)

# Using the global function for simple phase alignment
aligned_phase = align_phases(base_phase, target_phase)
```

#### Configuration Options:

- `lock_strength`: Strength of phase locking (0.0-1.0, default: 0.7)
- `phase_tolerance`: Tolerance for phase differences (default: 0.1)
- `max_history_length`: Maximum length of history to maintain (default: 1000)

### 3. CoherenceGate

The `CoherenceGate` class filters patterns based on coherence metrics:

```python
from utchs.core.coherence_gate import CoherenceGate, is_pattern_coherent

# Using the class
cg = CoherenceGate(config={'base_threshold': 0.7})
filtered_patterns = cg.filter_patterns(patterns, recursion_order=3)

# Using the global function for simple coherence check
is_coherent = is_pattern_coherent(pattern_data, recursion_order=3)
```

#### Configuration Options:

- `base_threshold`: Base coherence threshold (default: 0.7)
- `recursion_factor`: Factor to reduce threshold per recursion level (default: 0.1)
- `min_threshold`: Minimum allowed threshold (default: 0.4)
- `enable_adaptive_threshold`: Enable adaptive thresholds (default: True)
- `phase_weight`: Weight for phase coherence (default: 0.5)
- `energy_weight`: Weight for energy coherence (default: 0.3)
- `temporal_weight`: Weight for temporal coherence (default: 0.2)

### 4. Enhanced MobiusTransformation

The `MobiusTransformation` class has been enhanced with recursion-aware correction terms:

```python
from utchs.mathematics.mobius import MobiusTransformation

# Create transformation with recursion order
mobius = MobiusTransformation(a, b, c, d, recursion_order=3)

# Get corrected parameters
a_corr, b_corr, c_corr, d_corr = mobius.get_corrected_parameters()

# Apply transformation
transformed = mobius.transform(z)
```

#### Configuration Options:

- `recursion_order`: Order of recursion (default: 1)
- `enable_correction`: Enable correction terms (default: True)
- `min_denominator`: Minimum denominator for stability (default: 1e-10)

## Integration with MetaPatternDetector

The enhancements are integrated into the `MetaPatternDetector` class:

```python
from utchs.core.meta_pattern_detector import MetaPatternDetector

# Create detector with enhancements enabled
detector = MetaPatternDetector(config={
    'enable_phase_locking': True,
    'enable_coherence_gating': True,
    'max_recursion_order': 5
})

# Detect patterns with nonlinear correction
system_state = {
    'global_coherence': 0.85,
    'global_stability': 0.75,
    'energy_level': 0.8,
    'phase_recursion_depth': 3
}
results = detector.detect_all_meta_patterns(position_history, system_state=system_state)
```

## System Integration

The `UTCHSSystem` class has been updated to leverage these enhancements:

```python
from utchs.core.system import UTCHSSystem

# Create system
system = UTCHSSystem(config)

# Analyze meta-patterns with nonlinear corrections
analysis = system.analyze_meta_patterns(max_recursion_order=5)
```

## Example Usage

See `examples/nonlinear_recursion_example.py` for a complete demonstration of all the nonlinear recursion enhancements.

## Performance Considerations

1. **Computational Complexity**: The nonlinear enhancements may increase computational load, especially for large systems. Consider disabling some features for performance-critical applications.

2. **Memory Usage**: The TorsionalPhaseLock and CoherenceGate classes maintain history which can use additional memory. Adjust `max_history_length` if memory usage is a concern.

3. **Numeric Stability**: The Möbius corrections include safeguards for numeric stability. If you encounter instability, adjust the `min_denominator` parameter.

## Validation

The enhancements include validation in `utchs/utils/validation_registry.py` to ensure type safety and valid parameter ranges. Use the registry to validate your custom configurations:

```python
from utchs.utils.validation_registry import validation_registry

# Validate parameters
validation_registry.validate_module_data("RecursionScaling", config)
``` 