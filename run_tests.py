#!/usr/bin/env python3
"""
Convenient test runner script for P&ID processing pipeline.
"""

import sys
import subprocess
from pathlib import Path


def run_tests(test_type: str = "all", verbose: bool = True) -> int:
    """
    Run pytest with different configurations.

    Args:
        test_type: Type of tests to run ('all', 'unit', 'integration', 'config', 'parser', 'preprocess')
        verbose: Whether to run in verbose mode

    Returns:
        Exit code from pytest
    """
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    # Add test selection based on markers
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "config":
        cmd.extend(["-m", "config"])
    elif test_type == "parser":
        cmd.extend(["-m", "parser"])
    elif test_type == "preprocess":
        cmd.extend(["-m", "preprocess"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])
    elif test_type != "all":
        print(f"Unknown test type: {test_type}")
        print(
            "Available types: all, unit, integration, config, parser, preprocess, slow"
        )
        return 1

    # Add the tests directory
    cmd.append("tests/")

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point for test runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run tests for P&ID processing pipeline"
    )
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=[
            "all",
            "unit",
            "integration",
            "config",
            "parser",
            "preprocess",
            "slow",
        ],
        help="Type of tests to run (default: all)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Run in quiet mode")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting (requires pytest-cov)",
    )

    args = parser.parse_args()

    if args.coverage:
        # Add coverage options
        import os

        os.environ["PYTEST_ADDOPTS"] = "--cov=src --cov-report=html --cov-report=term"

    exit_code = run_tests(args.test_type, verbose=not args.quiet)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
