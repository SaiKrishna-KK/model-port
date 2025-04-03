# Contributing to ModelPort

Thank you for your interest in contributing to ModelPort! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate when interacting with other contributors. We aim to maintain a welcoming and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please report it by creating a new issue. When reporting a bug, please include:

1. A clear and descriptive title
2. Steps to reproduce the bug
3. Expected behavior
4. Actual behavior
5. Environment details (OS, Python version, etc.)
6. Any relevant logs or error messages

### Feature Requests

We welcome suggestions for new features. To request a feature:

1. Check if the feature has already been requested
2. Create a new issue with a clear description of the feature
3. Explain why this feature would be useful to ModelPort users

### Pull Requests

We welcome pull requests for bug fixes, features, and improvements. To submit a pull request:

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes
4. Add or update tests as necessary
5. Update documentation as necessary
6. Submit a pull request

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/SaiKrishna-KK/model-port.git
   cd model-port
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   pip install -r requirements.txt
   ```

3. Set up pre-commit hooks (recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Style

We follow standard Python style conventions:

- Use 4 spaces for indentation
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use meaningful variable and function names
- Write docstrings for all functions, classes, and modules
- Add type hints where appropriate

## Testing

Before submitting a pull request, please run the tests locally:

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_export.py

# Run tests with Docker (recommended for TVM-related changes)
docker build -t modelport-test -f Dockerfile.final .
docker run --rm modelport-test
```

## Documentation

Please update documentation when adding or modifying features:

1. Update the relevant parts of the documentation in `docs/`
2. Update docstrings for any new or modified functions/classes
3. Add examples if applicable

## Versioning

We use [Semantic Versioning](https://semver.org/) for releases:

- MAJOR version for incompatible API changes
- MINOR version for adding functionality in a backwards-compatible manner
- PATCH version for backwards-compatible bug fixes

## License

By contributing to ModelPort, you agree that your contributions will be licensed under the project's MIT license. 