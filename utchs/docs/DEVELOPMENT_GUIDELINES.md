# UTCHS Development Guidelines

## Table of Contents
1. [Code Organization](#code-organization)
2. [Project Structure](#project-structure)
3. [Coding Standards](#coding-standards)
4. [Documentation Standards](#documentation-standards)
5. [Version Control](#version-control)
6. [Testing Guidelines](#testing-guidelines)
7. [Performance Considerations](#performance-considerations)
8. [Validation Standards](#validation-standards)
9. [Naming Conventions](#naming-conventions)
10. [Dependency Management](#dependency-management)

## Code Organization

### Directory Structure
```
utchs/
├── docs/                  # Documentation
│   ├── api/              # API documentation
│   ├── examples/         # Example documentation
│   └── guides/           # Development guides
├── src/                  # Source code
│   ├── core/            # Core functionality
│   ├── math/            # Mathematical operations
│   ├── fields/          # Field implementations
│   ├── visualization/   # Visualization tools
│   └── utils/           # Utility functions
├── tests/               # Test suite
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── benchmarks/     # Performance benchmarks
├── examples/            # Example scripts
├── scripts/             # Utility scripts
├── config/             # Configuration files
└── requirements/       # Dependency management
    ├── base.txt       # Base requirements
    ├── dev.txt        # Development requirements
    └── test.txt       # Test requirements
```

### Module Organization

1. **Core Modules** (`core/`)
   - Each core concept should be in its own module
   - Modules should be focused and single-purpose
   - Dependencies between modules should be explicit
   - Example: `position.py`, `cycle.py`, `structure.py`

2. **Mathematical Modules** (`math/`)
   - Pure mathematical operations
   - No dependencies on other UTCHS modules
   - Should be easily testable
   - Example: `mobius.py`, `torsion.py`

3. **Field Modules** (`fields/`)
   - Field implementations
   - Clear separation between different field types
   - Consistent interface across field types
   - Example: `phase_field.py`, `energy_field.py`

4. **Visualization Modules** (`visualization/`)
   - Separate visualization logic from computation
   - Support multiple visualization backends
   - Consistent interface for all visualizations
   - Example: `field_vis.py`, `torus_vis.py`

## Project Structure

### File Naming Conventions

1. **Python Files**
   - Use lowercase with underscores
   - Descriptive and specific names
   - Examples:
     - `phase_field.py`
     - `mobius_transformation.py`
     - `field_visualizer.py`

2. **Test Files**
   - Prefix with `test_`
   - Match name of module being tested
   - Examples:
     - `test_phase_field.py`
     - `test_mobius_transformation.py`

3. **Configuration Files**
   - Use lowercase with underscores
   - Include environment if specific
   - Examples:
     - `default_config.yaml`
     - `development_config.yaml`

### Import Structure

1. **Standard Library Imports**
   ```python
   import os
   import sys
   from typing import Dict, List, Optional
   ```

2. **Third-Party Imports**
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   ```

3. **Local Imports**
   ```python
   from utchs.core.position import Position
   from utchs.math.mobius import MobiusTransformation
   ```

## Coding Standards

### General Guidelines

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use type hints consistently
   - Maximum line length: 88 characters
   - Use meaningful variable names
   - Write self-documenting code

2. **Function Design**
   - Single responsibility principle
   - Maximum function length: 50 lines
   - Clear input/output contracts
   - Proper error handling
   - Example:
   ```python
   def calculate_phase_circulation(
       field: np.ndarray,
       position: Tuple[int, int, int]
   ) -> float:
       """
       Calculate phase circulation around a point.
       
       Args:
           field: Complex-valued field array
           position: (i, j, k) position in the field
           
       Returns:
           float: Phase circulation in radians
           
       Raises:
           FieldError: If calculation fails
       """
   ```

3. **Class Design**
   - Clear class hierarchy
   - Encapsulation of implementation details
   - Consistent interface design
   - Example:
   ```python
   class PhaseField:
       """Complex-valued field with phase information."""
       
       def __init__(self, grid_size: Tuple[int, int, int]):
           """Initialize phase field."""
           
       @property
       def singularities(self) -> List[Dict]:
           """Get phase singularities in the field."""
           
       def update(self, transformation: MobiusTransformation) -> None:
           """Update field using transformation."""
   ```

## Documentation Standards

### Docstring Format

1. **Module Docstrings**
   ```python
   """
   Phase field implementation for UTCHS framework.
   
   This module provides functionality for managing and manipulating
   complex-valued phase fields in the UTCHS framework.
   """
   ```

2. **Class Docstrings**
   ```python
   class PhaseField:
       """
       Complex-valued field with phase information.
       
       This class manages a 3D complex-valued field that represents
       phase information in the UTCHS framework.
       
       Attributes:
           field: Complex-valued field array
           grid_size: Size of the field grid
           singularities: List of phase singularities
       """
   ```

3. **Method Docstrings**
   ```python
   def find_singularities(self) -> None:
       """
       Find phase singularities in the field.
       
       This method identifies points in the field where the phase
       is undefined or has a topological charge.
       
       Returns:
           None
           
       Raises:
           FieldError: If singularity detection fails
       """
   ```

## Validation Standards

### Using the Validation Registry

1. **Basic Validation**
   ```python
   from utchs.utils.validation_registry import validation_registry
   
   # Type validation
   validation_registry.validate("type", value, expected_type=int)
   
   # Range validation
   validation_registry.validate("range", value, min_value=0, max_value=100)
   
   # Array validation
   validation_registry.validate("array_shape", array, expected_shape=(10, 10))
   validation_registry.validate("array_dtype", array, expected_dtype=np.float64)
   ```

2. **Custom Validation**
   ```python
   def validate_custom(value: Any, **kwargs) -> None:
       """Custom validation function."""
       if not condition:
           raise UTCHSValidationError("Validation failed")
           
   validation_registry.register_validator("custom", validate_custom)
   validation_registry.validate("custom", value, **kwargs)
   ```

### Validation Best Practices

1. **Input Validation**
   - Validate all public method inputs
   - Use appropriate validation rules
   - Provide clear error messages
   - Handle validation errors gracefully

2. **Type Checking**
   - Use type hints consistently
   - Validate types at runtime
   - Use mypy for static type checking
   - Document type constraints

3. **Range Validation**
   - Validate numeric ranges
   - Check array shapes and types
   - Validate string lengths
   - Verify dictionary keys

## Naming Conventions

### UTCHS Data Dictionary

1. **Core Structural Classes**
   - `Position`: Fundamental unit in the UTCHS framework. Represents a point in the system with specific properties including spatial location, energy, phase, and relationships. Never use alternative names like "point" or "node".
   - `Cycle`: Collection of exactly 13 positions that form a complete rotation. Always use "Cycle" for this concept, not "Rotation" or "Round".
   - `ScaledStructure`: Collection of cycles with scaling transformations. Use this name consistently, not just "Structure".
   - `Torus`: Highest level organizational unit containing scaled structures. Always use "Torus" for this concept.

2. **Field Classes**
   - `PhaseField`: Complex-valued field representing phase dynamics. Handles phase evolution through Möbius transformations.
   - `EnergyField`: Real-valued field representing energy distribution. Distinct from phase field despite similar grid structure.
   - `TorsionField`: Field representing geometric torsion. While related to phase, serves a different mathematical purpose.
   
   Note: All field classes share a common grid-based structure but serve distinct purposes. Use the specific field type name, not generic terms like "grid" or "array".

3. **Visualization Classes**
   - `FieldVisualizer`: Base class for visualizing any field type.
   - `TorusVisualizer`: Specific to torus structure visualization.
   
   Note: Visualizer classes should always end with "Visualizer" suffix.

4. **Common Terms and Their Usage**
   - "position" (noun): A specific point in the UTCHS hierarchy
   - "field" (noun): A grid-based distribution of values
   - "cycle" (noun): A complete set of 13 positions
   - "structure" (noun): A scaled collection of cycles
   - "torus" (noun): The highest level organizational unit
   - "phase" (noun): Angular value in complex plane
   - "energy" (noun): Scalar value representing system energy
   - "torsion" (noun): Geometric twist measurement

5. **Attribute Naming Patterns**
   - `number`: For position numbering (1-13)
   - `id`: For unique identifiers
   - `field`: For grid-based data
   - `grid_size`: For field dimensions
   - `spatial_location`: For 3D coordinates
   - `rotational_angle`: For angular positions
   - `energy_level`: For energy values
   - `phase`: For phase values

### Naming Consistency Rules

1. **Class Names**
   - Use the exact names from the data dictionary
   - Never create alternative names for core concepts
   - Suffix patterns:
     - `*Field` for field classes
     - `*Visualizer` for visualization classes
     - `*Error` for custom exceptions

2. **Variable Names**
   - Use full words from the data dictionary
   - Be explicit about the type in the name
   - Examples:
     ```python
     # Good
     current_position = cycle.get_current_position()
     phase_field = PhaseField(grid_size=(50, 50, 50))
     
     # Bad
     pos = cycle.get_pos()  # Too vague
     field = PhaseField(grid_size=(50, 50, 50))  # Type not clear
     ```

3. **Method Names**
   - Use consistent verbs for similar operations
   - Examples:
     - `calculate_*` for computational methods
     - `update_*` for state changes
     - `get_*` for retrievers
     - `find_*` for search operations

## Dependency Management

### Using the Dependency Analyzer

1. **Running the Analyzer**
   ```python
   from utchs.utils.dependency_analyzer import dependency_analyzer
   
   # Analyze a single file
   dependency_analyzer.analyze_file(file_path)
   
   # Analyze entire directory
   dependency_analyzer.analyze_directory(directory_path)
   
   # Generate report
   report = dependency_analyzer.generate_report()
   
   # Visualize dependencies
   dependency_analyzer.visualize_dependencies()
   ```

2. **Dependency Best Practices**
   - Avoid circular dependencies
   - Keep modules focused and small
   - Use dependency injection
   - Document dependencies clearly

3. **Refactoring Guidelines**
   - Extract shared functionality
   - Split large modules
   - Use interfaces for loose coupling
   - Follow dependency inversion principle

### Running Codebase Analysis

1. **Using the Analysis Script**
   ```bash
   python -m utchs.scripts.analyze_codebase
   ```

2. **Analysis Reports**
   - `analysis_results.json`: Complete analysis results
   - `naming_report.txt`: Naming convention violations
   - `dependency_report.txt`: Module dependencies
   - `dependency_graph.png`: Visual dependency graph

3. **Addressing Issues**
   - Fix validation issues first
   - Address naming violations
   - Resolve circular dependencies
   - Follow refactoring suggestions

## Version Control

### Git Workflow

1. **Branch Naming**
   - Feature branches: `feature/description`
   - Bug fixes: `fix/description`
   - Documentation: `docs/description`
   - Performance: `perf/description`

2. **Commit Messages**
   - Clear and descriptive
   - Reference issue numbers
   - Follow conventional commits
   - Example:
     ```
     feat(phase_field): add singularity detection
     
     - Implement phase circulation calculation
     - Add singularity detection algorithm
     - Add tests for new functionality
     
     Closes #123
     ```

3. **Pull Requests**
   - Clear description of changes
   - Reference related issues
   - Include test coverage
   - Request reviews from maintainers

## Testing Guidelines

### Test Organization

1. **Unit Tests**
   - One test file per module
   - Clear test names
   - Isolated test cases
   - Example:
   ```python
   def test_phase_circulation():
       """Test phase circulation calculation."""
       field = PhaseField((10, 10, 10))
       circulation = field._calculate_phase_circulation((5, 5, 5))
       assert abs(circulation) <= 2 * np.pi
   ```

2. **Integration Tests**
   - Test module interactions
   - Test complete workflows
   - Example:
   ```python
   def test_field_update_workflow():
       """Test complete field update workflow."""
       field = PhaseField((10, 10, 10))
       transform = MobiusTransformation(1, 0, 0, 1)
       field.update(transform)
       assert field.is_valid()
   ```

## Performance Considerations

### Optimization Guidelines

1. **Algorithm Design**
   - Use vectorized operations
   - Minimize memory allocations
   - Profile critical paths
   - Example:
   ```python
   # Good
   field = np.exp(1j * phases)
   
   # Bad
   field = np.zeros_like(phases, dtype=complex)
   for i in range(phases.shape[0]):
       field[i] = np.exp(1j * phases[i])
   ```

2. **Memory Management**
   - Use in-place operations
   - Clear large objects when done
   - Monitor memory usage
   - Example:
   ```python
   # Good
   field *= np.exp(1j * phases)
   
   # Bad
   field = field * np.exp(1j * phases)
   ```

3. **Parallel Processing**
   - Use multiprocessing for CPU-bound tasks
   - Use threading for I/O-bound tasks
   - Example:
   ```python
   def process_field_chunk(chunk):
       """Process a chunk of the field."""
       return np.fft.fft2(chunk)
   
   with Pool() as pool:
       chunks = np.array_split(field, cpu_count())
       results = pool.map(process_field_chunk, chunks)
   ``` 