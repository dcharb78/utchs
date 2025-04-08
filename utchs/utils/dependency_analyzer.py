"""Dependency analyzer for the UTCHS framework.

This module provides tools to analyze and manage module dependencies within the framework.
"""

import ast
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import logging
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

logger = logging.getLogger(__name__)

class DependencyAnalyzer:
    """Analyzes and manages module dependencies in the UTCHS framework."""
    
    def __init__(self):
        """Initialize the dependency analyzer."""
        self.dependency_graph = nx.DiGraph()
        self.module_info: Dict[str, Dict] = defaultdict(dict)
        
    def analyze_file(self, file_path: Path) -> None:
        """Analyze dependencies in a Python file.
        
        Args:
            file_path: Path to the Python file
        """
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
                
            module_name = self._get_module_name(file_path)
            self._analyze_imports(tree, module_name, file_path)
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            
    def _get_module_name(self, file_path: Path) -> str:
        """Get the module name from a file path.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Module name
        """
        # Convert path to module name
        parts = list(file_path.parts)
        if 'utchs' in parts:
            idx = parts.index('utchs')
            module_parts = parts[idx:-1]  # Exclude the file name
            return '.'.join(module_parts)
        return file_path.stem
        
    def _analyze_imports(self, tree: ast.AST, module_name: str, file_path: Path) -> None:
        """Analyze import statements in an AST.
        
        Args:
            tree: AST to analyze
            module_name: Name of the current module
            file_path: Path to the file being analyzed
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    self._add_dependency(module_name, name.name, file_path)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self._add_dependency(module_name, node.module, file_path)
                    
    def _add_dependency(self, module_name: str, dependency: str, file_path: Path) -> None:
        """Add a dependency to the graph.
        
        Args:
            module_name: Name of the current module
            dependency: Name of the dependency
            file_path: Path to the file being analyzed
        """
        # Add nodes if they don't exist
        if module_name not in self.dependency_graph:
            self.dependency_graph.add_node(module_name)
            
        if dependency not in self.dependency_graph:
            self.dependency_graph.add_node(dependency)
            
        # Add edge
        self.dependency_graph.add_edge(module_name, dependency)
        
        # Store file path
        self.module_info[module_name]['file_path'] = file_path
        
    def analyze_directory(self, directory: Path) -> None:
        """Analyze dependencies in all Python files in a directory.
        
        Args:
            directory: Directory to analyze
        """
        for file_path in directory.rglob("*.py"):
            self.analyze_file(file_path)
            
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the graph.
        
        Returns:
            List of cycles, where each cycle is a list of module names
        """
        return list(nx.simple_cycles(self.dependency_graph))
        
    def get_module_dependencies(self, module_name: str) -> Tuple[Set[str], Set[str]]:
        """Get direct dependencies of a module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Tuple of (imported_modules, imported_by_modules)
        """
        if module_name not in self.dependency_graph:
            return set(), set()
            
        imported = set(self.dependency_graph.successors(module_name))
        imported_by = set(self.dependency_graph.predecessors(module_name))
        
        return imported, imported_by
        
    def get_dependency_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find the shortest dependency path between two modules.
        
        Args:
            source: Source module name
            target: Target module name
            
        Returns:
            List of module names in the path, or None if no path exists
        """
        try:
            return nx.shortest_path(self.dependency_graph, source, target)
        except nx.NetworkXNoPath:
            return None
            
    def visualize_dependencies(self, output_path: Optional[Path] = None) -> None:
        """Visualize the dependency graph.
        
        Args:
            output_path: Path to save the visualization, or None to display
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.dependency_graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.dependency_graph,
            pos,
            node_color='lightblue',
            node_size=2000
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.dependency_graph,
            pos,
            edge_color='gray',
            arrows=True
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.dependency_graph,
            pos,
            font_size=8
        )
        
        plt.title("Module Dependencies")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
            
    def generate_report(self) -> str:
        """Generate a report of the dependency analysis.
        
        Returns:
            Formatted report string
        """
        report = ["Dependency Analysis Report", "=" * 30, ""]
        
        # Report circular dependencies
        cycles = self.find_circular_dependencies()
        if cycles:
            report.append("Circular Dependencies:")
            for cycle in cycles:
                report.append(f"  {' -> '.join(cycle)} -> {cycle[0]}")
            report.append("")
            
        # Report module dependencies
        report.append("Module Dependencies:")
        for module in sorted(self.dependency_graph.nodes()):
            imported, imported_by = self.get_module_dependencies(module)
            
            report.append(f"\n{module}:")
            if imported:
                report.append("  Imports:")
                for dep in sorted(imported):
                    report.append(f"    - {dep}")
            if imported_by:
                report.append("  Imported by:")
                for dep in sorted(imported_by):
                    report.append(f"    - {dep}")
                    
        return "\n".join(report)
        
    def suggest_refactoring(self) -> List[str]:
        """Suggest refactoring to improve dependency structure.
        
        Returns:
            List of refactoring suggestions
        """
        suggestions = []
        
        # Check for circular dependencies
        cycles = self.find_circular_dependencies()
        if cycles:
            suggestions.append(
                "Found circular dependencies. Consider extracting shared functionality "
                "into a separate module or using dependency injection."
            )
            
        # Check for modules with too many dependencies
        for module in self.dependency_graph.nodes():
            deps = len(list(self.dependency_graph.successors(module)))
            if deps > 10:
                suggestions.append(
                    f"Module '{module}' has {deps} dependencies. Consider splitting "
                    "into smaller, more focused modules."
                )
                
        # Check for modules with too many dependents
        for module in self.dependency_graph.nodes():
            deps = len(list(self.dependency_graph.predecessors(module)))
            if deps > 10:
                suggestions.append(
                    f"Module '{module}' is imported by {deps} other modules. Consider "
                    "if this indicates tight coupling that should be reduced."
                )
                
        return suggestions

# Create a global instance
dependency_analyzer = DependencyAnalyzer() 