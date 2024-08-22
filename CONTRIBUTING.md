# Contributing to AmpSuite

We welcome contributions to AmpSuite! Every bit helps, and we appreciate your effort. Here's how you can contribute:

## Ways to Contribute

### 1. File an Issue

If you've found a bug or have a feature request, please [open an issue](https://github.inl.gov/richard-alfaro-diaz-lanl/AmpSuite/issues) on our GitHub repository.

#### Bug Reports

When filing a bug report, please include:

- Detailed steps to reproduce the bug
- Your operating system name and version
- Any details about your local setup that might be helpful in troubleshooting

#### Feature Requests

When submitting a feature request:

- Explain in detail how it would work
- Keep the scope as narrow as possible to make it easier to implement
- Remember that this is a volunteer-driven project, and contributions are welcome!

### 2. Submit a Pull Request

We appreciate code contributions! Here's how to submit a pull request:

1. Fork the `AmpSuite` repo on GitHub (if you don't have collaborator permissions).
2. Clone your fork locally:
   ```
   git clone git@github.com:your_username/AmpSuite.git
   ```
3. Create a branch for local development:
   ```
   git checkout -b name-of-your-bugfix-or-feature
   ```
4. Make your changes and commit them:
   ```
   git add .
   git commit -m "Your detailed description of your changes"
   ```
5. Push your branch to GitHub:
   ```
   git push origin name-of-your-bugfix-or-feature
   ```
6. Submit a pull request through the GitHub website.

### 3. Improve Documentation

Good documentation is crucial for any project. You can help by:

- Fixing typos or clarifying existing documentation
- Adding examples or use cases
- Writing tutorials or how-to guides

## Development Setup

To set up `AmpSuite` for local development:

1. Install the package in editable mode:
   ```
   pip install -e .
   ```
2. Install development dependencies:
   ```
   pip install -r requirements_dev.txt
   ```

## Code Style

We follow the PEP 8 style guide for Python code. Please ensure your code adheres to this standard.

## Running Tests (TO-DO)

Before submitting a pull request, please run the test suite to check that your changes don't break existing functionality:

```
pytest tests/
```

## Questions?

If you have any questions about contributing, feel free to ask in an issue or reach out to the maintainers directly.

Thank you for your interest in improving AmpSuite!
