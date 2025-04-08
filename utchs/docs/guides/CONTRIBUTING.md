# Contributing to UTCHS

Thank you for your interest in contributing to the UTCHS framework! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/utchs.git
   cd utchs
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements/dev.txt
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests:
   ```bash
   pytest
   ```
4. Run type checking:
   ```bash
   mypy .
   ```
5. Run linting:
   ```bash
   black .
   isort .
   pylint utchs
   ```
6. Commit your changes:
   ```bash
   git commit -m "feat: your feature description"
   ```
7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
8. Create a pull request

## Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all public functions/classes
- Keep functions focused and small
- Write tests for new features
- Update documentation as needed

## Testing

- Write unit tests for new features
- Ensure all tests pass
- Add integration tests for complex features
- Use pytest for testing
- Aim for high test coverage

## Documentation

- Update docstrings for modified code
- Add examples for new features
- Update README if needed
- Add API documentation for new modules
- Keep the changelog updated

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the CHANGELOG.md with details of changes
3. The PR will be merged once you have the sign-off of at least one maintainer

## Questions?

Feel free to open an issue for any questions about contributing. 