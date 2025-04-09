# UTCHS Meta Pattern Detector Testing Guide

This document outlines the testing approach for the UTCHS Meta Pattern Detector, which identifies recursive 3-6-9 patterns that generate complete 13D systems across multiple scales.

## Overview

The Meta Pattern Detector identifies and tracks recursive 3-6-9 patterns that generate entire new 13D systems at higher scales. The key function is to detect how the base 3-6-9 pattern (positions 3, 6, 9) generates a complete new 13D system at cycle 6, which then repeats this pattern to generate another system at cycle 12, and so on.

## What We're Testing

1. **Meta-Position Calculation**: Verify the calculation of cycle numbers corresponding to meta-positions at different recursion orders.
2. **Meta-Pattern Detection**: Test detection of meta-patterns at different recursion orders.
3. **Dimensional System Detection**: Test detection of complete 13D systems emerging at key cycles.
4. **Pattern Propagation Analysis**: Test analysis of how patterns propagate between recursion levels.
5. **Prediction Capabilities**: Test prediction of higher-order patterns and new 13D systems.

## Running the Tests

You can run the tests using the specialized test runner script:

```bash
python run_meta_pattern_tests.py
```

Or using pytest directly:

```bash
pytest -xvs tests/test_meta_pattern_detector.py
```

## Test Data

The tests use a combination of:

1. **Synthetic Data**: Automatically generated test data with known patterns.
2. **Cached Data**: For certain tests, data is cached in the `tests/data` directory.

## Expected Behavior

- Pattern detection should correctly identify 3-6-9 patterns at different scales.
- Meta-position calculations should follow the formula:
  - Meta₍ₙ₎(3) = Cycle(3 × 2ⁿ⁻¹)
  - Meta₍ₙ₎(6) = Cycle(6 × 2ⁿ⁻¹)
  - Meta₍ₙ₎(9) = Cycle(9 × 2ⁿ⁻¹)
- System detection should identify complete 13D systems at cycles 6, 12, 24, etc.
- Cross-scale correlations should show relationships between patterns at different recursion orders.

## Interpreting Results

Test results should show successful detection of:

- Meta-position 3 at cycles 3, 6, 12, 24, 48...
- Meta-position 6 at cycles 6, 12, 24, 48, 96...
- Meta-position 9 at cycles 9, 18, 36, 72, 144...

Complete 13D systems should be detected at cycles:
- 1st order: Base system (positions 1-13)
- 2nd order: Cycle 6
- 3rd order: Cycle 12
- 4th order: Cycle 24
- And so on...

## Troubleshooting

Common issues:

1. **Import Errors**: Ensure the meta pattern detector module is accessible in your Python path.
2. **Missing Dependencies**: Install required dependencies with `pip install numpy scipy pytest`.
3. **Mock Data Issues**: If tests fail due to mock data, delete the cached data in `tests/data` and run again.

## Next Steps

As the implementation evolves to fully reflect the understanding of 13D system generation, we'll need to:

1. Update tests to reflect the rename from MetaPatternDetector to CompleteSystemDetector
2. Add tests for system-to-system relationship analysis
3. Add visual verification tests for nested 13D system visualization 