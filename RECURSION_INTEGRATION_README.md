# UTCHS Recursion Integration

This document explains how to integrate recursion tracking, phase-locking, coherence gating, and meta-pattern detection into the main UTCHSSystem.

## Overview

The recursion integration enhances the UTCHSSystem with the following capabilities:

1. **Recursion Tracking**: Track positions across multiple recursion levels (octaves)
2. **Phase-Locking**: Align phases between recursion levels to prevent destructive interference
3. **Coherence Gating**: Filter patterns based on multi-dimensional coherence metrics
4. **Meta-Pattern Detection**: Identify recursive 3-6-9 patterns across multiple scales

## Components

### RecursionTracker Singleton

The `RecursionTracker` class maintains a record of positions as they transition through different recursive levels. It is implemented as a singleton to ensure all components access the same tracking instance.

### Phase-Locking System

The `TorsionalPhaseLock` class aligns phases between different recursion levels to prevent destructive interference between 13D systems that emerge at different recursion orders.

### Coherence Gate

The `CoherenceGate` class filters detected patterns based on coherence metrics, preventing noise amplification at higher recursion orders.

### Meta-Pattern Detector

The `MetaPatternDetector` class identifies recursive 3-6-9 patterns that emerge at cycle 6, where cycle 6 becomes a "meta-position 3" in a higher-order pattern.

### UTCHSSystemIntegrator

The `UTCHSSystemIntegrator` class ties all these components together and integrates them with the main UTCHSSystem.

## Quick Start

### Basic Integration

```python
from utchs.core.system import UTCHSSystem
from utchs.core.system_integration import integrate_recursion_tracking, create_default_configuration

# Create a UTCHSSystem
config = {
    'grid_size': (32, 32, 32),
    'grid_spacing': 0.25,
    'history_length': 1000
}
system = UTCHSSystem(config)

# Create default recursion tracking configuration
recursion_config = create_default_configuration()

# Integrate recursion tracking
integrator = integrate_recursion_tracking(system, recursion_config)

# Run simulation with enhanced recursion tracking
states = system.run_simulation(1000)

# Generate a report
system.generate_recursion_report("recursion_report.txt")

# When done, you can detach the components (optional)
integrator.detach_components()
```

### Accessing Recursion Components

After integration, you can access the recursion components through the UTCHSSystem:

```python
# Get recursion tracker instance
recursion_tracker = system.get_recursion_tracker()

# Get transition analyzer instance
transition_analyzer = system.get_transition_analyzer()

# Get fractal analyzer instance
fractal_analyzer = system.get_fractal_analyzer()

# Get meta-pattern detector instance
meta_pattern_detector = system.get_meta_pattern_detector()
```

### Analyzing Recursion Patterns

You can analyze recursion patterns in the system:

```python
# Get transitions
transitions = recursion_tracker.get_recursion_transitions()

# Detect phi resonances
phi_resonance = transition_analyzer.detect_phi_resonances()

# Calculate fractal properties
fractal_dimension = fractal_analyzer.calculate_fractal_dimension()
self_similarity = fractal_analyzer.calculate_self_similarity()

# Detect meta-patterns
meta_pattern = meta_pattern_detector.detect_meta_patterns(
    recursion_tracker.position_history
)
```

## Configuration Options

### Main Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `tracking_interval` | 1 | How often to track positions (in ticks) |
| `analysis_interval` | 100 | How often to run analysis (in ticks) |
| `visualization_interval` | 500 | How often to create visualizations (in ticks) |
| `output_dir` | 'recursion_output' | Output directory for reports and visualizations |

### RecursionTracker Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `max_history_length` | 1000 | Maximum number of positions to store in history |
| `max_recursion_depth` | 7 | Maximum recursion depth to track |

### Meta-Pattern Detector Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `correlation_threshold` | 0.7 | Threshold for correlation detection |
| `phase_coherence_threshold` | 0.6 | Threshold for phase coherence |
| `energy_pattern_threshold` | 0.65 | Threshold for energy pattern detection |
| `max_recursion_order` | 5 | Maximum recursion order to detect |
| `enable_phase_locking` | True | Enable phase-locking between recursion levels |
| `enable_coherence_gating` | True | Enable coherence gating for pattern filtering |

### Phase-Locking Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `lock_strength` | 0.7 | Strength of phase-locking (0.0-1.0) |
| `phase_tolerance` | 0.1 | Tolerance for phase differences |

### Coherence Gate Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `base_threshold` | 0.7 | Base threshold for coherence filtering |
| `recursion_factor` | 0.1 | How much to reduce threshold per recursion level |
| `min_threshold` | 0.4 | Minimum allowed threshold |
| `enable_adaptive_threshold` | True | Enable adaptive thresholds based on history |

## Examples

See `examples/recursion_integration_example.py` for a complete example of how to use the recursion integration.

## Testing

The integration includes comprehensive tests in `tests/test_system_integration.py`. You can run the tests using pytest:

```bash
pytest tests/test_system_integration.py
```

## Benefits of Integration

1. **Comprehensive Tracking**: Track positions across multiple recursion levels
2. **Enhanced Analysis**: Analyze recursive patterns and phi resonances
3. **Mathematical Precision**: Ensure mathematical correctness with phase-locking
4. **Noise Reduction**: Filter out noise with coherence gating
5. **Meta-Pattern Detection**: Identify emerging recursive patterns
6. **Visual Insights**: Create visualizations of recursion effects

## Implementation Details

The integration uses a non-invasive method hooking approach:

1. The original methods of UTCHSSystem are preserved
2. New methods are added as wrappers around the original methods
3. The system can function without recursion components if needed
4. Components can be detached to restore the original system

This ensures compatibility with existing code while providing enhanced capabilities. 