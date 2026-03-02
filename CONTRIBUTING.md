# Contributing to FusionMind 4.0

Thank you for your interest in FusionMind 4.0.

## Current Status

FusionMind 4.0 is under active development with patent filings in progress (PF1–PF7). Contributions are welcome but subject to IP review.

## How to Contribute

### Bug Reports

Open an issue with:
- Python version and OS
- Minimal reproduction steps
- Expected vs actual behavior
- Full traceback

### Feature Requests

Open an issue describing:
- The use case
- How it relates to existing patent families
- Proposed implementation approach

### Code Contributions

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for any new functionality
4. Ensure all 56+ tests pass: `pytest tests/ -v`
5. Submit a pull request

### Code Standards

- Python 3.10+ compatible
- Type hints for public APIs
- Docstrings for all public classes and functions
- Tests required for all new functionality
- No external dependencies beyond `numpy`, `scipy`, `networkx` for core

## Development Setup

```bash
git clone https://github.com/mladen1312/FusionMind-4-CausalPlasma.git
cd FusionMind-4-CausalPlasma
pip install -e ".[full,dev]"
pytest tests/ -v
```

## IP Notice

By contributing, you agree that your contributions may be included in patent filings. All contributors will be acknowledged.

## Contact

Dr. Mladen Mester — via GitHub Issues
