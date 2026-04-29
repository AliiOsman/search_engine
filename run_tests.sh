#!/bin/bash

echo "======================================"
echo "  Quote Search Engine — Test Runner"
echo "======================================"

# Install coverage if not already installed
echo ""
echo ">>> Installing coverage..."
pip install coverage -q

# Run all tests with coverage
echo ""
echo ">>> Running all tests..."
echo ""
python -m coverage run -m unittest discover -s tests/ -p "test_*.py" -v

# Print coverage report
echo ""
echo "======================================"
echo "  Coverage Report"
echo "======================================"
python -m coverage report --show-missing

# Generate HTML report
echo ""
echo ">>> Generating HTML coverage report..."
python -m coverage html
echo ""
echo ">>> Done! Open htmlcov/index.html in your browser for the visual report."