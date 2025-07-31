#!/usr/bin/env python3
"""
Environment validation script for Snowflake2025 Hex AI project.

This script validates that the development environment is properly configured
and provides clear guidance for fixing any issues. It's designed to help
coding agents understand the required setup.

Usage:
    python scripts/validate_environment.py [--fix] [--verbose]
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse


class EnvironmentValidator:
    """Validates the development environment for the Hex AI project."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        self.issues = []
        self.warnings = []
        self.successes = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with appropriate formatting."""
        if level == "ERROR":
            print(f"❌ ERROR: {message}")
        elif level == "WARNING":
            print(f"⚠️  WARNING: {message}")
        elif level == "SUCCESS":
            print(f"✅ SUCCESS: {message}")
        elif self.verbose:
            print(f"ℹ️  INFO: {message}")
    
    def check_virtual_environment(self) -> bool:
        """Check if we're running in the correct virtual environment."""
        venv_path = os.environ.get("VIRTUAL_ENV", "")
        
        if not venv_path:
            self.issues.append(
                "No virtual environment detected. "
                "You must activate the 'hex_ai_env' virtual environment first."
            )
            return False
        
        if "hex_ai_env" not in venv_path:
            self.issues.append(
                f"Wrong virtual environment detected: {venv_path}\n"
                f"Expected environment containing 'hex_ai_env'"
            )
            return False
        
        self.successes.append(f"Virtual environment: {venv_path}")
        return True
    
    def check_python_path(self) -> bool:
        """Check if PYTHONPATH includes the project root."""
        python_path = os.environ.get("PYTHONPATH", "")
        project_root_str = str(self.project_root)
        
        # Check if project root is in Python's sys.path (which includes PYTHONPATH)
        if str(self.project_root) not in sys.path:
            if not python_path:
                self.warnings.append(
                    "PYTHONPATH is not set. This may cause import issues.\n"
                    "Consider setting: export PYTHONPATH=."
                )
            else:
                self.issues.append(
                    f"Project root not in Python path: {project_root_str}\n"
                    f"Current PYTHONPATH: {python_path}\n"
                    f"Python sys.path: {sys.path[:3]}..."  # Show first few entries
                )
            return False
        
        self.successes.append(f"PYTHONPATH includes project root: {project_root_str}")
        return True
    
    def check_project_structure(self) -> bool:
        """Check that essential project files and directories exist."""
        required_files = [
            "requirements.txt",
            "hex_ai/__init__.py",
            "scripts/__init__.py",
            "tests/__init__.py",
        ]
        
        required_dirs = [
            "hex_ai",
            "scripts", 
            "tests",
            "data",
            "checkpoints",
        ]
        
        missing_files = []
        missing_dirs = []
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        for dir_path in required_dirs:
            if not (self.project_root / dir_path).is_dir():
                missing_dirs.append(dir_path)
        
        if missing_files:
            self.issues.append(f"Missing required files: {missing_files}")
        
        if missing_dirs:
            self.issues.append(f"Missing required directories: {missing_dirs}")
        
        if not missing_files and not missing_dirs:
            self.successes.append("Project structure is complete")
            return True
        
        return False
    
    def check_dependencies(self) -> bool:
        """Check that required Python packages are installed."""
        required_packages = [
            "torch",
            "numpy", 
            "scipy",
            "tqdm",
            "psutil",
            "flask",
            "flask_cors",
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                importlib.import_module(package)
                if self.verbose:
                    self.log(f"Package {package} is available")
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.issues.append(f"Missing required packages: {missing_packages}")
            return False
        
        self.successes.append("All required packages are installed")
        return True
    
    def check_imports(self) -> bool:
        """Test that key project modules can be imported."""
        test_imports = [
            "hex_ai",
            "hex_ai.models",
            "hex_ai.training_orchestration",
            "hex_ai.inference.model_wrapper",
        ]
        
        failed_imports = []
        
        for module in test_imports:
            try:
                importlib.import_module(module)
                if self.verbose:
                    self.log(f"Module {module} imports successfully")
            except ImportError as e:
                failed_imports.append(f"{module}: {e}")
        
        if failed_imports:
            self.issues.append(f"Failed imports: {failed_imports}")
            return False
        
        self.successes.append("All key modules can be imported")
        return True
    
    def check_gpu_availability(self) -> bool:
        """Check GPU availability and configuration."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                self.successes.append(f"CUDA GPU available: {gpu_name} (count: {gpu_count})")
                return True
            elif torch.backends.mps.is_available():
                self.successes.append("MPS (Apple Silicon) GPU available")
                return True
            else:
                self.warnings.append("No GPU acceleration available - training will be slower")
                return True
        except ImportError:
            self.warnings.append("PyTorch not available - cannot check GPU")
            return False
    
    def print_setup_instructions(self):
        """Print clear setup instructions for coding agents."""
        print("\n🔧 ENVIRONMENT SETUP:")
        print("1. source hex_ai_env/bin/activate")
        print("2. export PYTHONPATH=.")
        print("3. python scripts/validate_environment.py")
        print("\nSee AGENT_GUIDANCE.md for more details.")
    
    def print_fix_suggestions(self):
        """Print suggestions for fixing detected issues."""
        if not self.issues and not self.warnings:
            return
        
        print("\n" + "="*80)
        print("🔧 FIX SUGGESTIONS")
        print("="*80)
        
        if self.issues:
            print("\n❌ CRITICAL ISSUES TO FIX:")
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. {issue}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS TO ADDRESS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning}")
        
        print("\n" + "="*80)
    
    def run_all_checks(self) -> bool:
        """Run all environment checks and return overall success."""
        print("🔍 Validating Snowflake2025 development environment...")
        print()
        
        checks = [
            ("Virtual Environment", self.check_virtual_environment),
            ("Project Structure", self.check_project_structure),
            ("Dependencies", self.check_dependencies),
            ("PYTHONPATH", self.check_python_path),
            ("Module Imports", self.check_imports),
            ("GPU Availability", self.check_gpu_availability),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            try:
                if not check_func():
                    all_passed = False
            except Exception as e:
                self.issues.append(f"{check_name} check failed with error: {e}")
                all_passed = False
        
        print("\n" + "="*80)
        print("📊 VALIDATION RESULTS")
        print("="*80)
        
        if self.successes:
            print("\n✅ SUCCESSES:")
            for success in self.successes:
                print(f"  • {success}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if self.issues:
            print("\n❌ ISSUES:")
            for issue in self.issues:
                print(f"  • {issue}")
        
        if all_passed:
            print("\n🎉 All checks passed! Environment is properly configured.")
        else:
            print("\n💥 Environment validation failed. Please fix the issues above.")
        
        return all_passed


def main():
    """Main entry point for the environment validator."""
    parser = argparse.ArgumentParser(
        description="Validate the Snowflake2025 development environment"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--show-instructions",
        action="store_true", 
        help="Show setup instructions for coding agents"
    )
    
    args = parser.parse_args()
    
    validator = EnvironmentValidator(verbose=args.verbose)
    
    if args.show_instructions:
        validator.print_setup_instructions()
        return
    
    success = validator.run_all_checks()
    
    if not success:
        validator.print_fix_suggestions()
        validator.print_setup_instructions()
        sys.exit(1)
    
    print("\n🚀 Ready to work on Snowflake2025!")


if __name__ == "__main__":
    main() 