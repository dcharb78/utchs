"""Naming convention checker for the UTCHS framework.

This module provides tools to enforce consistent naming conventions across the framework.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import ast
import logging

logger = logging.getLogger(__name__)

class NamingChecker:
    """Enforces naming conventions for the UTCHS framework."""
    
    # Naming patterns
    CLASS_PATTERN = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
    METHOD_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')
    VARIABLE_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')
    CONSTANT_PATTERN = re.compile(r'^[A-Z][A-Z0-9_]*$')
    PRIVATE_PATTERN = re.compile(r'^_[a-z][a-z0-9_]*$')
    
    # Reserved words that should not be used
    RESERVED_WORDS = {
        'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
        'del', 'elif', 'else', 'except', 'exec', 'finally', 'for',
        'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'None',
        'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with',
        'yield', 'True', 'False'
    }
    
    # Common abbreviations to avoid
    AVOIDED_ABBREVIATIONS = {
        'calc': 'calculate',
        'coord': 'coordinate',
        'ctx': 'context',
        'def': 'define',
        'doc': 'document',
        'err': 'error',
        'func': 'function',
        'idx': 'index',
        'info': 'information',
        'init': 'initialize',
        'len': 'length',
        'loc': 'location',
        'msg': 'message',
        'num': 'number',
        'obj': 'object',
        'param': 'parameter',
        'pos': 'position',
        'proc': 'process',
        'ref': 'reference',
        'temp': 'temporary',
        'val': 'value',
        'var': 'variable'
    }
    
    # Common acceptable abbreviations
    ACCEPTABLE_ABBREVIATIONS = {
        'dir',      # directory
        'str',      # string
        'int',      # integer
        'dict',     # dictionary
        'arr',      # array
        'cfg',      # config
        'max',      # maximum
        'min',      # minimum
        'id',       # identifier
        'args',     # arguments
        'kwargs',   # keyword arguments
        'env',      # environment
        'src',      # source
        'dst',      # destination
        'tmp',      # temporary
        'db',       # database
        'lib',      # library
        'pkg',      # package
        'mod',      # module
        'fn',       # function
        'cls',      # class
        'img',      # image
        'doc',      # documentation
        'auth',     # authentication
        'admin',    # administrator
        'config',   # configuration
        'utils',    # utilities
        'params',   # parameters
        'stats',    # statistics
        'deps',     # dependencies
    }
    
    def __init__(self):
        """Initialize the naming checker."""
        self.violations: List[Tuple[str, str, str]] = []
        
    def check_file(self, file_path: Path) -> List[Tuple[str, str, str]]:
        """Check naming conventions in a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of (line_number, name, violation_message) tuples
        """
        self.violations = []
        
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
                
            self._check_node(tree, file_path)
            return self.violations
            
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {str(e)}")
            return []
            
    def _check_node(self, node: ast.AST, file_path: Path) -> None:
        """Recursively check an AST node for naming violations.
        
        Args:
            node: AST node to check
            file_path: Path to the file being checked
        """
        if isinstance(node, ast.ClassDef):
            self._check_class_name(node, file_path)
            
        elif isinstance(node, ast.FunctionDef):
            self._check_function_name(node, file_path)
            
        elif isinstance(node, ast.Name):
            self._check_variable_name(node, file_path)
            
        # Recursively check child nodes
        for child in ast.iter_child_nodes(node):
            self._check_node(child, file_path)
            
    def _check_class_name(self, node: ast.ClassDef, file_path: Path) -> None:
        """Check a class name for violations.
        
        Args:
            node: Class definition node
            file_path: Path to the file being checked
        """
        name = node.name
        
        # Check pattern
        if not self.CLASS_PATTERN.match(name):
            self.violations.append((
                str(node.lineno),
                name,
                f"Class name '{name}' should use PascalCase"
            ))
            
        # Check for reserved words
        if name.lower() in self.RESERVED_WORDS:
            self.violations.append((
                str(node.lineno),
                name,
                f"Class name '{name}' is a reserved word"
            ))
            
        # Check for avoided abbreviations
        for abbr, full in self.AVOIDED_ABBREVIATIONS.items():
            if abbr in name.lower():
                self.violations.append((
                    str(node.lineno),
                    name,
                    f"Class name '{name}' contains avoided abbreviation '{abbr}' (use '{full}' instead)"
                ))
                
    def _check_function_name(self, node: ast.FunctionDef, file_path: Path) -> None:
        """Check a function name for violations.
        
        Args:
            node: Function definition node
            file_path: Path to the file being checked
        """
        name = node.name
        
        # Skip special methods
        if name.startswith('__') and name.endswith('__'):
            return
            
        # Check pattern
        if not self.METHOD_PATTERN.match(name):
            self.violations.append((
                str(node.lineno),
                name,
                f"Function name '{name}' should use snake_case"
            ))
            
        # Check for reserved words
        if name.lower() in self.RESERVED_WORDS:
            self.violations.append((
                str(node.lineno),
                name,
                f"Function name '{name}' is a reserved word"
            ))
            
        # Check for avoided abbreviations
        for abbr, full in self.AVOIDED_ABBREVIATIONS.items():
            if abbr in name.lower():
                self.violations.append((
                    str(node.lineno),
                    name,
                    f"Function name '{name}' contains avoided abbreviation '{abbr}' (use '{full}' instead)"
                ))
                
    def _check_variable_name(self, node: ast.Name, file_path: Path) -> None:
        """Check a variable name for violations.
        
        Args:
            node: Name node
            file_path: Path to the file being checked
        """
        name = node.id
        
        # Skip if it's a built-in name, special name, or type hint
        if (name in dir(__builtins__) or 
            (name.startswith('__') and name.endswith('__')) or
            name in {'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple', 'Any', 'Optional', 'Union', 'List', 'Dict', 'Set', 'Tuple'}):
            return
            
        # Check pattern based on context
        if isinstance(node.ctx, ast.Store):  # Assignment
            if not (self.VARIABLE_PATTERN.match(name) or self.CONSTANT_PATTERN.match(name)):
                self.violations.append((
                    str(node.lineno),
                    name,
                    f"Variable name '{name}' should use snake_case or UPPER_CASE for constants"
                ))
                
        # Check for reserved words
        if name.lower() in self.RESERVED_WORDS:
            self.violations.append((
                str(node.lineno),
                name,
                f"Variable name '{name}' is a reserved word"
            ))
            
        # Check for avoided abbreviations
        for abbr, full in self.AVOIDED_ABBREVIATIONS.items():
            if abbr in name.lower():
                # Skip if the name is a type hint or acceptable abbreviation
                if name in {'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple'} or any(acc in name.lower() for acc in self.ACCEPTABLE_ABBREVIATIONS):
                    continue
                self.violations.append((
                    str(node.lineno),
                    name,
                    f"Variable name '{name}' contains avoided abbreviation '{abbr}' (use '{full}' instead)"
                ))
                
    def check_directory(self, directory: Path) -> Dict[str, List[Tuple[str, str, str]]]:
        """Check naming conventions in all Python files in a directory.
        
        Args:
            directory: Directory to check
            
        Returns:
            Dictionary mapping file paths to lists of violations
        """
        results = {}
        
        for file_path in directory.rglob("*.py"):
            violations = self.check_file(file_path)
            if violations:
                results[str(file_path)] = violations
                
        return results
        
    def generate_report(self, results: Dict[str, List[Tuple[str, str, str]]]) -> str:
        """Generate a report of naming convention violations.
        
        Args:
            results: Dictionary of violations by file
            
        Returns:
            Formatted report string
        """
        if not results:
            return "No naming convention violations found."
            
        report = ["Naming Convention Violations Report", "=" * 30, ""]
        
        for file_path, violations in results.items():
            report.append(f"File: {file_path}")
            report.append("-" * len(f"File: {file_path}"))
            
            for line, name, message in violations:
                report.append(f"Line {line}: {message}")
                
            report.append("")
            
        return "\n".join(report)

# Create a global instance
naming_checker = NamingChecker() 