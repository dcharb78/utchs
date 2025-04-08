Contributing to UTCHS
===================

Thank you for your interest in contributing to UTCHS! This guide will help you get started.

Development Setup
---------------

1. Fork the repository
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/yourusername/utchs.git
      cd utchs

3. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install development dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

Code Style
---------

UTCHS follows PEP 8 style guidelines. We use the following tools to maintain code quality:

- black: Code formatting
- isort: Import sorting
- mypy: Type checking
- pylint: Code analysis

Run the tools:

.. code-block:: bash

   # Format code
   black utchs
   isort utchs

   # Type checking
   mypy utchs

   # Code analysis
   pylint utchs

Testing
-------

We use pytest for testing. Run the tests:

.. code-block:: bash

   pytest

Write tests for new features:

1. Create test files in the appropriate test directory
2. Use pytest fixtures for setup
3. Follow the existing test patterns
4. Aim for good test coverage

Documentation
------------

We use Sphinx for documentation. Build the docs:

.. code-block:: bash

   cd docs
   make html

When adding new features:

1. Update API documentation
2. Add docstrings to all new code
3. Update relevant guides
4. Add examples if applicable

Pull Request Process
------------------

1. Create a new branch:

   .. code-block:: bash

      git checkout -b feature/my-feature

2. Make your changes
3. Run tests and linting
4. Commit your changes:

   .. code-block:: bash

      git add .
      git commit -m "Description of changes"

5. Push to your fork:

   .. code-block:: bash

      git push origin feature/my-feature

6. Create a pull request

Code Review
----------

All code changes require review. The review process:

1. Automated checks must pass
2. At least one maintainer must approve
3. All comments must be addressed
4. Documentation must be updated

Release Process
-------------

1. Update version in setup.py
2. Update CHANGELOG.md
3. Create a release branch
4. Run full test suite
5. Build documentation
6. Create a release tag
7. Push to PyPI

Getting Help
-----------

- Open an issue for bugs
- Use discussions for questions
- Join our community chat

Thank you for contributing! 