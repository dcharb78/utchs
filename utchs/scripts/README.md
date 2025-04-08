# UTCHS Codebase Analysis

This directory contains scripts for analyzing the UTCHS codebase and generating reports.

## Analysis Script

The main analysis script (`analyze_codebase.py`) performs comprehensive analysis of the codebase, checking for:

- Validation patterns and usage
- Naming convention violations
- Module dependencies and potential issues

### Usage

Run the analysis script from the project root:

```bash
python -m utchs.scripts.analyze_codebase
```

### Output

The script generates reports in a timestamped directory under `analysis_reports/`:

- `analysis_results.json`: Complete analysis results in JSON format
- `naming_report.txt`: Detailed report of naming convention violations
- `dependency_report.txt`: Module dependency information
- `dependency_graph.png`: Visual representation of module dependencies

### Interpreting Results

#### Validation Analysis
- Lists all registered validators
- (TODO) Shows validation usage patterns across the codebase

#### Naming Convention Analysis
- Groups violations by type (class names, method names, etc.)
- Provides line numbers and context for each violation
- Suggests corrections for violations

#### Dependency Analysis
- Shows module dependency statistics
- Identifies circular dependencies
- Provides refactoring suggestions to improve code organization

### Best Practices

1. Run the analysis regularly as part of your development workflow
2. Address critical issues (e.g., circular dependencies) promptly
3. Use the reports to guide refactoring efforts
4. Keep track of improvements over time

## Contributing

When adding new analysis features:

1. Add new analysis functions to `analyze_codebase.py`
2. Update the report generation in `generate_reports()`
3. Document new features in this README
4. Add appropriate tests for new analysis functionality 