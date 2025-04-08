# UTCHS Framework Implementation

This repository contains a Python implementation of the Enhanced Unified Toroidal-Crystalline Harmonic System (UTCHS) framework, a comprehensive theoretical model that integrates concepts of toroidal geometry, crystalline structures, harmonic systems, and phase recursion.

## Overview

The UTCHS framework represents a paradigm shift from traditional models, focusing on:

1. **Dynamic Phase Relationships** rather than static structures
2. **Recursive Transformations** rather than linear scaling
3. **Interconnected Fields** rather than isolated components
4. **Phase-Based Understanding** rather than purely geometric perspectives

The core insight of the framework is: "It's not the shape. It's the recursion of phase through the shape."

## Key Components

The implementation is organized into several core modules:

### Core Structural Components

- **Position**: Fundamental unit with properties tied to its number (1-13)
- **Cycle**: Collection of 13 positions forming a complete cycle
- **Structure**: Collection of cycles (typically 7) with φ-scaling
- **Torus**: Collection of structures forming a complete toroidal geometry

### Mathematical Framework

- **Möbius Transformations**: Core mathematical structure for phase evolution
- **Torsion Field**: Tensor describing how the phase field "twists" around each point
- **Phase Field**: Complex-valued field evolving through Möbius transformations
- **Energy Field**: Field with non-local feedback mechanisms

### Special Elements

- **Prime Torsional Anchors**: Prime number positions (1, 2, 3, 5, 7, 11, 13) that provide stability
- **Vortex Points**: Special positions (3, 6, 9) that serve as energy vortices
- **Phase Singularities**: Points where phase is undefined, creating vortex-like structures
- **Resonant Channels**: Energy pathways between resonant positions

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/utchs-framework.git
   cd utchs-framework
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Simulation

Run a basic simulation with the default configuration:

```bash
python basic_simulation.py
```

### Configuration Options

The system can be configured using the `default_config.yaml` file or by specifying a custom configuration file:

```bash
python basic_simulation.py --config custom_config.yaml
```

### Visualization

Generate visualizations of the simulation results:

```bash
python basic_simulation.py --visualize
```

### Analysis

Perform detailed analysis of the system state:

```bash
python basic_simulation.py --analysis
```

### Extended Run

Run a longer simulation with more ticks:

```bash
python basic_simulation.py --ticks 1000
```

### Loading from Checkpoint

Resume a simulation from a saved checkpoint:

```bash
python basic_simulation.py --checkpoint checkpoints/utchs_checkpoint_500.npz
```

## Framework Structure

```
utchs/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── config.py
│   └── default_config.yaml
├── core/
│   ├── __init__.py
│   ├── position.py
│   ├── cycle.py
│   ├── structure.py
│   ├── torus.py
│   └── system.py
├── math/
│   ├── __init__.py
│   ├── mobius.py
│   ├── torsion.py
│   ├── recursion.py
│   └── primes.py
├── fields/
│   ├── __init__.py
│   ├── phase_field.py
│   └── energy_field.py
├── visualization/
│   ├── __init__.py
│   ├── field_vis.py
│   └── torus_vis.py
├── utils/
│   ├── __init__.py
│   └── storage.py
└── examples/
    └── basic_simulation.py
```

## Examples

### Creating a Simple UTCHS System

```python
from utchs.config import load_config
from utchs.core.system import UTCHSSystem

# Load configuration
config = load_config()

# Create system
system = UTCHSSystem(config)

# Run simulation for 100 ticks
states = system.run_simulation(100)

# Analyze system state
analysis = system.analyze_system_state()
print(f"Global coherence: {analysis['stability_analysis']['current_stability']:.4f}")
```

### Visualizing Phase Fields

```python
from utchs.visualization.field_vis import FieldVisualizer

# Create visualizer
field_vis = FieldVisualizer()

# Visualize phase field
field_vis.visualize_phase_slice(system.phase_field)

# Visualize energy field
field_vis.visualize_energy_slice(system.energy_field)

# Create 3D visualization of phase singularities
field_vis.visualize_phase_singularities(system.phase_field)
```

### Visualizing Toroidal Structures

```python
from utchs.visualization.torus_vis import TorusVisualizer

# Create visualizer
torus_vis = TorusVisualizer()

# Visualize torus structure
torus_vis.visualize_torus(system.tori[0])

# Visualize position network
torus_vis.visualize_position_network(system)

# Visualize resonance network
torus_vis.visualize_resonance_network(system)
```

## Theory and Background

The UTCHS framework integrates several concepts from mathematics, physics, and information theory:

- **Toroidal Geometry**: Self-referential structures where inside and outside become relative
- **Harmonic Systems**: Resonant patterns based on musical ratios and phase relationships
- **Phase Recursion**: Self-referential phase dynamics creating complex emergent patterns
- **Torsion Fields**: Fields describing the twisting of space around energy centers
- **Golden Ratio (φ) Scaling**: Optimal scaling relationships appearing throughout the system

The framework has connections to various fields including quantum physics, music theory, information theory, and consciousness studies.

## Contributing

Contributions to the UTCHS framework are welcome! Please feel free to submit pull requests or open issues for discussion.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This implementation is based on theoretical work developed by researchers exploring complex recursive systems, phase dynamics, and toroidal geometries.
