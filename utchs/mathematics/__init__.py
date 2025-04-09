"""
UTCHS Math Module (DEPRECATED)

This module has been renamed to 'mathematics' to avoid conflicts with Python's built-in math module.
Please update your imports to use 'utchs.mathematics' instead.

For backward compatibility, this module still exists, but it will be removed in a future version.
"""

import warnings

warnings.warn(
    "The 'utchs.math' module has been renamed to 'utchs.mathematics' to avoid "
    "conflicts with Python's built-in 'math' module. Please update your imports. "
    "The old module will be maintained for backward compatibility but will be "
    "removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)
