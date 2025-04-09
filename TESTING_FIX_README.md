# UTCHS Testing Fix

This package provides a fix for the import conflict and validation function registration issues in the UTCHS framework tests.

## Issue Description

The UTCHS framework encountered two issues when running tests:

1. **Import Conflict**: The custom `utchs.math` package conflicts with Python's built-in `math` module, causing import errors.

2. **Missing Validation Function**: The validation registry was missing the `register_validation_function` implementation needed by the resonance detection system.

## Fix Components

The fix consists of several components:

1. **Import Conflict Resolver** (`utchs/tests/fix_imports.py`): Patches Python's import system to correctly handle the conflict between the built-in `math` module and the `utchs.math` package.

2. **Validation Utilities** (`utchs/utils/validation_utils.py`): Ensures all validation functions are properly registered with the validation registry.

3. **Test Environment Setup** (`setup_test_env.py`): Sets up the test environment with all required dependencies.

4. **Test Runner** (`run_tests.py`): Runs the tests with the fixes applied.

## How to Use

### Option 1: Using the Test Runner

The simplest way to run tests with the fix applied is to use the `run_tests.py` script:

```bash
python run_tests.py
```

This script:
- Checks for required dependencies
- Applies the import conflict fix
- Registers validation functions
- Runs the tests

### Option 2: Manual Setup

If you prefer to run tests manually:

1. First, set up the test environment:

```bash
python setup_test_env.py
```

2. Import the fix in your test file:

```python
# At the top of your test file
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utchs.tests.fix_imports import fix_path

# Import validation utils to register functions
import utchs.utils.validation_utils
```

3. Run your tests as usual:

```bash
cd utchs
python -m pytest tests/test_data_generation.py -v
```

## Understanding the Fix

### Import Conflict Resolution

The fix works by patching Python's import system to handle the special case of the `math` module. When code tries to import the built-in `math` module, our patch ensures it gets the correct one, not the local `utchs.math` package.

### Validation Function Registration

The validation utilities module registers all required validation functions with the validation registry, ensuring that functions like `validate_resonant_frequency` are available when needed by other components.

## Additional Notes

- The fix is designed to be non-invasive and can be removed once a more permanent solution is implemented.

- For development, consider using explicit imports like `from utchs.math import ...` rather than `import utchs.math` to avoid confusion.

- In the long term, consider renaming the `utchs.math` package to something more specific like `utchs.mathematics` to avoid the conflict entirely. 