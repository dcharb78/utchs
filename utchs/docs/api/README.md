# API Documentation

This directory contains the API documentation for the UTCHS framework. The documentation is generated using Sphinx and includes:

- Detailed API references for all modules
- Type hints and function signatures
- Usage examples
- Class hierarchies
- Module dependencies

## Building the Documentation

To build the API documentation:

```bash
cd docs
make html
```

The generated documentation will be available in `docs/_build/html/`.

## Documentation Structure

- `core/` - Core UTCHS components
- `math/` - Mathematical operations and transformations
- `fields/` - Field implementations
- `visualization/` - Visualization tools
- `utils/` - Utility functions

## Contributing

When adding new features or modifying existing ones:

1. Update the relevant module documentation
2. Add type hints to all functions and methods
3. Include usage examples
4. Document any breaking changes
5. Update the changelog 