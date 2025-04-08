#!/usr/bin/env python3
"""
UTCHS Codebase Analysis Script

This script runs various analyses on the UTCHS codebase to identify potential issues
and generate reports for validation, naming conventions, and dependencies.
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utchs.utils.validation_registry import validation_registry
from utchs.utils.naming_checker import naming_checker
from utchs.utils.dependency_analyzer import dependency_analyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_codebase() -> Dict:
    """
    Run comprehensive analysis of the codebase.
    
    Returns:
        Dict containing analysis results
    """
    logger.info("Starting codebase analysis...")
    
    # Get the root directory of the project
    root_dir = Path(__file__).parent.parent.parent
    utchs_dir = root_dir / "utchs"
    
    # Run all analyses
    results = {
        "validation": analyze_validation(),
        "naming": analyze_naming(utchs_dir),
        "dependencies": analyze_dependencies(utchs_dir)
    }
    
    logger.info("Codebase analysis completed")
    return results

def analyze_validation() -> Dict:
    """
    Analyze validation usage in the codebase.
    
    Returns:
        Dict containing validation analysis results
    """
    logger.info("Analyzing validation usage...")
    
    # Get all registered validators
    validators = validation_registry.list_validators()
    
    # TODO: Add analysis of validator usage in the codebase
    # This would require parsing Python files and looking for validation calls
    
    return {
        "registered_validators": list(validators),
        "validation_usage": {}  # To be implemented
    }

def analyze_naming(utchs_dir: Path) -> Dict:
    """
    Analyze naming conventions in the codebase.
    
    Args:
        utchs_dir: Path to the UTCHS source directory
        
    Returns:
        Dict containing naming analysis results
    """
    logger.info("Analyzing naming conventions...")
    
    # Check all Python files in the UTCHS directory
    violations = naming_checker.check_directory(utchs_dir)
    
    # Group violations by type
    grouped_violations = {}
    for file_path, file_violations in violations.items():
        for line, name, message in file_violations:
            violation_type = message.split(":")[0]
            if violation_type not in grouped_violations:
                grouped_violations[violation_type] = []
            grouped_violations[violation_type].append({
                "file": file_path,
                "line": line,
                "name": name,
                "message": message
            })
    
    return {
        "total_violations": sum(len(v) for v in violations.values()),
        "violations_by_type": grouped_violations
    }

def analyze_dependencies(utchs_dir: Path) -> Dict:
    """
    Analyze module dependencies in the codebase.
    
    Args:
        utchs_dir: Path to the UTCHS source directory
        
    Returns:
        Dict containing dependency analysis results
    """
    logger.info("Analyzing module dependencies...")
    
    # Analyze all Python files
    dependency_analyzer.analyze_directory(utchs_dir)
    
    # Get dependency information
    graph = dependency_analyzer.dependency_graph
    circular_deps = dependency_analyzer.find_circular_dependencies()
    
    # Get module statistics
    module_stats = {
        "total_modules": len(graph.nodes),
        "total_dependencies": len(graph.edges),
        "circular_dependencies": len(circular_deps)
    }
    
    return {
        "module_statistics": module_stats,
        "circular_dependencies": circular_deps,
        "suggestions": dependency_analyzer.suggest_refactoring()
    }

def generate_reports(results: Dict, output_dir: Path) -> None:
    """
    Generate analysis reports.
    
    Args:
        results: Analysis results
        output_dir: Directory to save reports
    """
    logger.info("Generating reports...")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete results as JSON
    results_file = output_dir / "analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate naming convention report
    naming_report = ["Naming Convention Violations Report", "=" * 30, ""]
    
    for violation_type, violations in results["naming"]["violations_by_type"].items():
        naming_report.append(f"\n{violation_type}:")
        naming_report.append("-" * len(violation_type))
        
        for violation in violations:
            naming_report.append(
                f"File: {violation['file']}, Line {violation['line']}: {violation['message']}"
            )
    
    with open(output_dir / "naming_report.txt", "w") as f:
        f.write("\n".join(naming_report))
    
    # Generate dependency report
    dep_report = dependency_analyzer.generate_report()
    with open(output_dir / "dependency_report.txt", "w") as f:
        f.write(dep_report)
    
    # Generate dependency graph visualization
    dependency_analyzer.visualize_dependencies(
        output_dir / "dependency_graph.png"
    )
    
    logger.info(f"Reports generated in {output_dir}")

def main():
    """Main entry point for the analysis script."""
    try:
        # Run analysis
        results = analyze_codebase()
        
        # Generate reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("analysis_reports") / timestamp
        generate_reports(results, output_dir)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 