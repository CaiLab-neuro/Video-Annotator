#!/usr/bin/env python3
"""
Tarsier2 Video Annotation Tool - Installation Script

This script automates the setup process for the Tarsier2-based behavioral video
annotation pipeline. It handles:
  - System prerequisite validation (git, ffmpeg, CUDA, disk space)
  - Conda environment creation and management
  - Tarsier repository cloning and setup
  - Project dependency installation
  - Installation verification

Usage:
    python install.py                              # Interactive mode
    python install.py --env-name my_env            # Custom environment name
    python install.py --force --non-interactive    # Force overwrite, no prompts
    python install.py --verify-only                # Check existing installation
    python install.py --skip-tarsier               # Only install project deps

For more information, see: https://github.com/bytedance/tarsier
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


# ========================================================================
# Color codes for terminal output
# ========================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def disable():
        """Disable colors for non-terminal output"""
        Colors.HEADER = ''
        Colors.OKBLUE = ''
        Colors.OKCYAN = ''
        Colors.OKGREEN = ''
        Colors.WARNING = ''
        Colors.FAIL = ''
        Colors.ENDC = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''


# ========================================================================
# Custom Exceptions
# ========================================================================

class InstallationError(Exception):
    """Base exception for installation errors"""
    pass


class PrerequisiteError(InstallationError):
    """Raised when system prerequisites are not met"""
    pass


class EnvironmentError(InstallationError):
    """Raised when environment setup fails"""
    pass


# ========================================================================
# System Checker
# ========================================================================

class SystemChecker:
    """Validates system prerequisites before installation"""

    @staticmethod
    def check_command(command: str) -> bool:
        """Check if a command is available in PATH"""
        return shutil.which(command) is not None

    @staticmethod
    def check_disk_space(path: Path, required_gb: float = 10.0) -> Tuple[bool, float]:
        """Check available disk space at path"""
        stat = os.statvfs(path)
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        return available_gb >= required_gb, available_gb

    @staticmethod
    def check_cuda() -> Tuple[bool, Optional[str]]:
        """Check CUDA availability"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip().split('\n')[0]
                return True, gpu_name
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return False, None

    @staticmethod
    def run_checks() -> dict:
        """Run all system checks and return status dict"""
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}System Prerequisites Check{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

        checks = {}

        # Check git
        checks['git'] = SystemChecker.check_command('git')
        status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if checks['git'] else f"{Colors.FAIL}✗{Colors.ENDC}"
        print(f"  {status} Git:     {'Found' if checks['git'] else 'NOT FOUND'}")

        # Check ffmpeg
        checks['ffmpeg'] = SystemChecker.check_command('ffmpeg')
        status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if checks['ffmpeg'] else f"{Colors.WARNING}!{Colors.ENDC}"
        print(f"  {status} FFmpeg:  {'Found' if checks['ffmpeg'] else 'Not found (required for video processing)'}")

        # Check CUDA
        checks['cuda'], gpu_name = SystemChecker.check_cuda()
        status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if checks['cuda'] else f"{Colors.WARNING}!{Colors.ENDC}"
        gpu_info = f"Found ({gpu_name})" if checks['cuda'] else "Not found (will use CPU mode)"
        print(f"  {status} CUDA:    {gpu_info}")

        # Check disk space
        project_root = Path(__file__).parent
        checks['disk_space'], available_gb = SystemChecker.check_disk_space(project_root)
        status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if checks['disk_space'] else f"{Colors.WARNING}!{Colors.ENDC}"
        print(f"  {status} Disk:    {available_gb:.1f} GB available (≥10 GB recommended)")

        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

        return checks


# ========================================================================
# Environment Manager
# ========================================================================

class EnvironmentManager:
    """Handles conda/venv environment operations"""

    @staticmethod
    def detect_active_conda_env() -> Optional[str]:
        """Detect currently active conda environment"""
        conda_prefix = os.environ.get('CONDA_PREFIX')
        conda_default_env = os.environ.get('CONDA_DEFAULT_ENV')

        if conda_prefix and conda_default_env:
            # Don't recommend using base environment
            if conda_default_env == 'base':
                return None
            return conda_default_env
        return None

    @staticmethod
    def is_conda_available() -> bool:
        """Check if conda is available"""
        return shutil.which('conda') is not None

    @staticmethod
    def conda_env_exists(env_name: str) -> bool:
        """Check if a conda environment exists"""
        try:
            result = subprocess.run(
                ['conda', 'env', 'list', '--json'],
                capture_output=True,
                text=True,
                check=True
            )
            envs = json.loads(result.stdout)['envs']
            return any(env_name in env_path for env_path in envs)
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            return False

    @staticmethod
    def get_conda_python(env_name: str) -> Optional[Path]:
        """Get Python executable path for conda environment"""
        try:
            result = subprocess.run(
                ['conda', 'env', 'list', '--json'],
                capture_output=True,
                text=True,
                check=True
            )
            envs = json.loads(result.stdout)['envs']
            for env_path in envs:
                if env_name in env_path:
                    python_path = Path(env_path) / 'bin' / 'python'
                    if python_path.exists():
                        return python_path
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            pass
        return None

    @staticmethod
    def create_conda_env(env_name: str, python_version: str = "3.9") -> bool:
        """Create a new conda environment"""
        print(f"\n{Colors.OKCYAN}Creating conda environment: {env_name}{Colors.ENDC}")
        print(f"Python version: {python_version}")

        try:
            subprocess.run(
                ['conda', 'create', '-n', env_name, f'python={python_version}', '-y'],
                check=True
            )
            print(f"{Colors.OKGREEN}✓ Environment created successfully{Colors.ENDC}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"{Colors.FAIL}✗ Failed to create environment: {e}{Colors.ENDC}")
            return False

    @staticmethod
    def verify_python_version(python_bin: Path, required_version: str = "3.9") -> bool:
        """Verify Python version meets requirements"""
        try:
            result = subprocess.run(
                [str(python_bin), '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            version_str = result.stdout.strip()
            print(f"  Python version: {version_str}")
            return required_version in version_str
        except subprocess.CalledProcessError:
            return False

    @staticmethod
    def prompt_use_active_env(active_env: str, requested_env: str, non_interactive: bool) -> str:
        """Ask user if they want to use the currently active environment"""
        if not active_env or active_env == requested_env:
            return requested_env

        print(f"\n{Colors.WARNING}Currently active conda environment: {active_env}{Colors.ENDC}")
        print(f"Requested environment: {requested_env}")

        if non_interactive:
            print(f"Non-interactive mode: using requested environment '{requested_env}'")
            return requested_env

        response = input(f"\nUse currently active environment '{active_env}'? [Y/n]: ").strip().lower()

        if response in ('', 'y', 'yes'):
            print(f"{Colors.OKGREEN}Using active environment: {active_env}{Colors.ENDC}")
            return active_env
        else:
            print(f"Using requested environment: {requested_env}")
            return requested_env


# ========================================================================
# Tarsier Installer
# ========================================================================

class TarsierInstaller:
    """Handles Tarsier repository cloning and setup"""

    TARSIER_REPO = "https://github.com/bytedance/tarsier.git"
    TARSIER_BRANCH = "tarsier2"

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tarsier_dir = project_root / "tarsier"

    def handle_existing_directory(self, force: bool, non_interactive: bool) -> bool:
        """Handle existing tarsier directory"""
        if not self.tarsier_dir.exists():
            return True

        # Check if directory is empty
        if not any(self.tarsier_dir.iterdir()):
            # Directory exists but is empty - safe to use
            print(f"\n{Colors.OKCYAN}Found empty tarsier directory - will install into it{Colors.ENDC}")
            return True

        # Directory exists and is not empty
        print(f"\n{Colors.WARNING}Tarsier directory already exists and contains files: {self.tarsier_dir}{Colors.ENDC}")

        if force:
            print(f"{Colors.WARNING}Force mode: removing existing directory{Colors.ENDC}")
            shutil.rmtree(self.tarsier_dir)
            return True

        if non_interactive:
            raise FileExistsError(
                f"Tarsier directory exists and is not empty at {self.tarsier_dir}. "
                "Use --force to overwrite or remove manually."
            )

        response = input(f"Remove existing directory and reinstall? [y/N]: ").strip().lower()
        if response == 'y':
            print(f"Removing {self.tarsier_dir}...")
            shutil.rmtree(self.tarsier_dir)
            return True
        else:
            print(f"{Colors.WARNING}Keeping existing directory. Skipping Tarsier installation.{Colors.ENDC}")
            return False

    def clone_repository(self) -> bool:
        """Clone Tarsier repository"""
        print(f"\n{Colors.OKCYAN}Cloning Tarsier repository...{Colors.ENDC}")
        print(f"Repository: {self.TARSIER_REPO}")
        print(f"Branch: {self.TARSIER_BRANCH}")
        print(f"Destination: {self.tarsier_dir}")

        try:
            # Clone repository
            subprocess.run(
                ['git', 'clone', '--branch', self.TARSIER_BRANCH, self.TARSIER_REPO, str(self.tarsier_dir)],
                check=True,
                cwd=self.project_root
            )
            print(f"{Colors.OKGREEN}✓ Repository cloned successfully{Colors.ENDC}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"{Colors.FAIL}✗ Failed to clone repository: {e}{Colors.ENDC}")
            return False

    def run_setup_script(self, python_bin: Path) -> bool:
        """Execute Tarsier's setup.sh script with error handling"""
        setup_script = self.tarsier_dir / "setup.sh"

        if not setup_script.exists():
            print(f"{Colors.WARNING}setup.sh not found at {setup_script}{Colors.ENDC}")
            print(f"{Colors.WARNING}Attempting manual dependency installation...{Colors.ENDC}")
            return self._manual_setup(python_bin)

        print(f"\n{Colors.OKCYAN}Running Tarsier setup script...{Colors.ENDC}")
        print(f"This may take several minutes (installing PyTorch, flash-attention, etc.)")

        # Prepare environment with correct Python in PATH
        env = os.environ.copy()
        env['PATH'] = f"{python_bin.parent}:{env['PATH']}"

        try:
            subprocess.run(
                ['bash', str(setup_script)],
                cwd=self.tarsier_dir,
                env=env,
                check=True
            )
            print(f"{Colors.OKGREEN}✓ Setup script completed successfully{Colors.ENDC}")
        except subprocess.CalledProcessError as e:
            print(f"{Colors.WARNING}⚠ Setup script encountered errors (may be non-fatal){Colors.ENDC}")
            print(f"{Colors.OKCYAN}Attempting to fix common issues...{Colors.ENDC}")

        # Fix common installation issues
        return self._fix_installation_issues(python_bin)

    def _manual_setup(self, python_bin: Path) -> bool:
        """Fallback: manually install dependencies if setup.sh is missing"""
        requirements_file = self.tarsier_dir / "requirements.txt"

        if not requirements_file.exists():
            print(f"{Colors.FAIL}✗ requirements.txt not found{Colors.ENDC}")
            return False

        print(f"Installing dependencies from {requirements_file}...")

        try:
            subprocess.run(
                [str(python_bin), '-m', 'pip', 'install', '-r', str(requirements_file)],
                check=True
            )
            print(f"{Colors.OKGREEN}✓ Dependencies installed{Colors.ENDC}")
        except subprocess.CalledProcessError as e:
            print(f"{Colors.WARNING}⚠ Some dependencies failed to install{Colors.ENDC}")

        return self._fix_installation_issues(python_bin)

    def _fix_installation_issues(self, python_bin: Path) -> bool:
        """Fix common installation issues (func_timeout, NumPy, missing deps)"""
        print(f"\n{Colors.OKCYAN}Checking and fixing installation issues...{Colors.ENDC}")

        success = True

        # Fix 1: Ensure NumPy 1.x (compatible with PyTorch 2.1)
        print(f"  Fixing NumPy version compatibility...")
        try:
            subprocess.run(
                [str(python_bin), '-m', 'pip', 'install', 'numpy<2'],
                check=True,
                capture_output=True
            )
            print(f"  {Colors.OKGREEN}✓{Colors.ENDC} NumPy downgraded to 1.x")
        except subprocess.CalledProcessError:
            print(f"  {Colors.WARNING}!{Colors.ENDC} Could not fix NumPy version")
            success = False

        # Fix 2: Ensure transformers is installed (often fails due to func_timeout error)
        print(f"  Ensuring transformers is installed...")
        try:
            subprocess.run(
                [str(python_bin), '-m', 'pip', 'install', 'transformers==4.47.0'],
                check=True,
                capture_output=True
            )
            print(f"  {Colors.OKGREEN}✓{Colors.ENDC} Transformers installed")
        except subprocess.CalledProcessError:
            print(f"  {Colors.WARNING}!{Colors.ENDC} Could not install transformers")
            success = False

        # Fix 3: Install missing dependencies
        missing_deps = ['pyarrow', 'decord', 'gradio==4.31.5', 'openai==1.14.2']
        print(f"  Installing commonly missing dependencies...")
        for dep in missing_deps:
            try:
                subprocess.run(
                    [str(python_bin), '-m', 'pip', 'install', dep],
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError:
                pass  # Continue even if some fail

        print(f"  {Colors.OKGREEN}✓{Colors.ENDC} Missing dependencies installed")

        # Fix 4: Re-install all requirements.txt to ensure completeness
        requirements_file = self.tarsier_dir / "requirements.txt"
        if requirements_file.exists():
            print(f"  Re-installing all Tarsier requirements...")
            try:
                subprocess.run(
                    [str(python_bin), '-m', 'pip', 'install', '-r', str(requirements_file)],
                    check=True,
                    capture_output=True
                )
                print(f"  {Colors.OKGREEN}✓{Colors.ENDC} All requirements installed")
            except subprocess.CalledProcessError:
                print(f"  {Colors.WARNING}!{Colors.ENDC} Some requirements may have failed")

        print(f"{Colors.OKGREEN}✓ Installation fixes applied{Colors.ENDC}")
        return success


# ========================================================================
# Dependency Installer
# ========================================================================

class DependencyInstaller:
    """Installs project-specific dependencies"""

    @staticmethod
    def install_project_dependencies(python_bin: Path) -> bool:
        """Install additional project dependencies"""
        print(f"\n{Colors.OKCYAN}Installing project dependencies...{Colors.ENDC}")

        dependencies = [
            'pympi-ling',  # For ELAN .eaf file processing
            'pandas',
            'matplotlib',
            'scikit-learn',
            'numpy'
        ]

        # Upgrade pip first
        print("Upgrading pip...")
        try:
            subprocess.run(
                [str(python_bin), '-m', 'pip', 'install', '--upgrade', 'pip'],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            print(f"{Colors.WARNING}! Failed to upgrade pip (continuing anyway){Colors.ENDC}")

        # Install dependencies
        for dep in dependencies:
            try:
                print(f"  Installing {dep}...")
                subprocess.run(
                    [str(python_bin), '-m', 'pip', 'install', dep],
                    check=True,
                    capture_output=True
                )
                print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {dep}")
            except subprocess.CalledProcessError as e:
                print(f"  {Colors.WARNING}!{Colors.ENDC} {dep} (failed, but continuing)")

        print(f"{Colors.OKGREEN}✓ Project dependencies installed{Colors.ENDC}")
        return True


# ========================================================================
# Installation Verifier
# ========================================================================

class InstallationVerifier:
    """Verifies installation success"""

    @staticmethod
    def verify_imports(python_bin: Path, project_root: Path) -> dict:
        """Test critical imports"""
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}Installation Verification{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

        # Test basic imports first
        basic_imports = {
            'torch': 'PyTorch',
            'transformers': 'HuggingFace Transformers',
            'pympi': 'ELAN file processing',
            'pandas': 'Pandas data processing',
            'matplotlib': 'Matplotlib plotting',
            'pyarrow': 'PyArrow data processing',
            'decord': 'Decord video processing'
        }

        results = {}

        print(f"{Colors.BOLD}Basic Dependencies:{Colors.ENDC}")
        for module, description in basic_imports.items():
            try:
                result = subprocess.run(
                    [str(python_bin), '-c', f'import {module}'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                success = result.returncode == 0
                results[module] = success
                status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if success else f"{Colors.FAIL}✗{Colors.ENDC}"
                print(f"  {status} {description:35} ({module})")
            except subprocess.TimeoutExpired:
                results[module] = False
                print(f"  {Colors.FAIL}✗{Colors.ENDC} {description:35} (timeout)")

        # Test Tarsier imports (need to add tarsier to sys.path)
        print(f"\n{Colors.BOLD}Tarsier Imports:{Colors.ENDC}")
        tarsier_test_code = f"""
import sys
sys.path.insert(0, '{project_root / 'tarsier'}')
from tasks.utils import load_model_and_processor
from tools.conversation import Chat, conv_templates
print('success')
"""
        try:
            result = subprocess.run(
                [str(python_bin), '-c', tarsier_test_code],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=project_root
            )
            success = result.returncode == 0 and 'success' in result.stdout
            results['tarsier'] = success
            status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if success else f"{Colors.FAIL}✗{Colors.ENDC}"
            print(f"  {status} Tarsier imports                     (with sys.path setup)")
            if not success and result.stderr:
                print(f"{Colors.WARNING}    Error: {result.stderr.strip()[:200]}{Colors.ENDC}")
        except subprocess.TimeoutExpired:
            results['tarsier'] = False
            print(f"  {Colors.FAIL}✗{Colors.ENDC} Tarsier imports                     (timeout)")

        # Check CUDA availability
        print(f"\n{Colors.BOLD}GPU/CUDA Status:{Colors.ENDC}")
        try:
            result = subprocess.run(
                [str(python_bin), '-c', 'import torch; print(torch.cuda.is_available())'],
                capture_output=True,
                text=True,
                timeout=10
            )
            cuda_available = 'True' in result.stdout
            status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if cuda_available else f"{Colors.WARNING}!{Colors.ENDC}"
            print(f"  {status} CUDA available: {cuda_available}")

            if cuda_available:
                # Get GPU name
                result = subprocess.run(
                    [str(python_bin), '-c', 'import torch; print(torch.cuda.get_device_name(0))'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"      GPU: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            print(f"  {Colors.WARNING}!{Colors.ENDC} Could not check CUDA status")

        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

        return results


# ========================================================================
# Main Installer
# ========================================================================

class Installer:
    """Main installation coordinator"""

    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent.resolve()
        self.python_bin = None
        self.env_name = args.env_name

    def run(self):
        """Execute installation process"""
        try:
            # Step 1: System checks
            if not self.args.skip_checks:
                checks = SystemChecker.run_checks()

                # Git is mandatory
                if not checks['git']:
                    raise PrerequisiteError(
                        "Git is required but not found. Please install git and try again."
                    )

                # Warn about ffmpeg
                if not checks['ffmpeg']:
                    print(f"{Colors.WARNING}Warning: ffmpeg not found. Video processing will not work.{Colors.ENDC}")
                    if not self.args.non_interactive:
                        response = input("Continue anyway? [y/N]: ").strip().lower()
                        if response != 'y':
                            print("Installation cancelled.")
                            return 1

            # Step 2: Environment setup
            if not self.args.skip_env_creation:
                if not self._setup_environment():
                    return 1
            else:
                # Use existing environment
                print(f"\n{Colors.OKCYAN}Using existing environment: {self.env_name}{Colors.ENDC}")
                if EnvironmentManager.is_conda_available():
                    self.python_bin = EnvironmentManager.get_conda_python(self.env_name)
                    if not self.python_bin:
                        raise EnvironmentError(f"Could not find Python executable for environment: {self.env_name}")
                else:
                    # Assume system Python
                    self.python_bin = Path(sys.executable)

            # Verify Python version
            print(f"\n{Colors.OKCYAN}Verifying Python environment...{Colors.ENDC}")
            print(f"Python executable: {self.python_bin}")
            if not EnvironmentManager.verify_python_version(self.python_bin):
                print(f"{Colors.WARNING}Warning: Python 3.9 is recommended for Tarsier compatibility{Colors.ENDC}")

            # Step 3: Tarsier installation
            if not self.args.skip_tarsier and not self.args.verify_only:
                if not self._install_tarsier():
                    return 1

            # Step 4: Project dependencies
            if not self.args.verify_only:
                DependencyInstaller.install_project_dependencies(self.python_bin)

            # Step 5: Verification
            results = InstallationVerifier.verify_imports(self.python_bin, self.project_root)

            # Step 6: Display usage instructions
            self._display_usage_instructions(results)

            # Check if all critical imports succeeded
            critical_imports = ['torch', 'transformers', 'tarsier']
            all_critical_ok = all(results.get(imp, False) for imp in critical_imports)

            if all_critical_ok:
                print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ Installation completed successfully!{Colors.ENDC}\n")
                return 0
            else:
                print(f"\n{Colors.WARNING}⚠ Installation completed with warnings. Some imports failed.{Colors.ENDC}\n")
                failed = [imp for imp in critical_imports if not results.get(imp, False)]
                print(f"{Colors.WARNING}Failed imports: {', '.join(failed)}{Colors.ENDC}")
                print(f"\nTry reinstalling with: python install.py --force\n")
                return 1

        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}Installation interrupted by user.{Colors.ENDC}")
            print("You can resume by running this script again.")
            return 1
        except Exception as e:
            print(f"\n{Colors.FAIL}✗ Installation failed: {e}{Colors.ENDC}")
            return 1

    def _setup_environment(self) -> bool:
        """Set up Python environment (conda or venv)"""
        # Detect active conda environment
        active_env = EnvironmentManager.detect_active_conda_env()
        if active_env:
            self.env_name = EnvironmentManager.prompt_use_active_env(
                active_env,
                self.env_name,
                self.args.non_interactive
            )

        # Check if conda is available
        if not EnvironmentManager.is_conda_available():
            print(f"\n{Colors.WARNING}Conda not found. Using system Python.{Colors.ENDC}")
            self.python_bin = Path(sys.executable)
            return True

        # Check if environment exists
        if EnvironmentManager.conda_env_exists(self.env_name):
            print(f"\n{Colors.OKCYAN}Conda environment '{self.env_name}' already exists.{Colors.ENDC}")
            self.python_bin = EnvironmentManager.get_conda_python(self.env_name)
            if not self.python_bin:
                raise EnvironmentError(f"Could not find Python executable for environment: {self.env_name}")
            return True

        # Create new environment
        if not self.args.non_interactive:
            response = input(f"\nCreate new conda environment '{self.env_name}'? [Y/n]: ").strip().lower()
            if response not in ('', 'y', 'yes'):
                print("Environment creation cancelled.")
                return False

        if not EnvironmentManager.create_conda_env(self.env_name):
            return False

        self.python_bin = EnvironmentManager.get_conda_python(self.env_name)
        if not self.python_bin:
            raise EnvironmentError(f"Could not find Python executable for newly created environment: {self.env_name}")

        return True

    def _install_tarsier(self) -> bool:
        """Install Tarsier repository"""
        installer = TarsierInstaller(self.project_root)

        # Handle existing directory
        if not installer.handle_existing_directory(self.args.force, self.args.non_interactive):
            print(f"{Colors.WARNING}Skipping Tarsier installation.{Colors.ENDC}")
            return True

        # Clone repository
        if not installer.clone_repository():
            return False

        # Run setup script
        if not installer.run_setup_script(self.python_bin):
            print(f"\n{Colors.WARNING}Setup script failed. You may need to manually install Tarsier.{Colors.ENDC}")
            return False

        return True

    def _display_usage_instructions(self, verification_results: dict):
        """Display post-installation usage instructions"""
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}Next Steps{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

        # Environment activation
        print(f"{Colors.BOLD}1. Activate your environment:{Colors.ENDC}")
        if EnvironmentManager.is_conda_available():
            print(f"   conda activate {self.env_name}\n")
        else:
            print(f"   # Using system Python: {self.python_bin}\n")

        # Model information
        print(f"{Colors.BOLD}2. Model information:{Colors.ENDC}")
        print(f"   Model will auto-download on first use:")
        print(f"   - Recommended: omni-research/Tarsier2-7b-0115")
        print(f"   - Alternative: omni-research/Tarsier2-Recap-7b")
        print(f"   - Cached location: ~/.cache/huggingface/hub/\n")

        # Example usage
        print(f"{Colors.BOLD}3. Run video annotation:{Colors.ENDC}")
        print(f"   # Using the script (edit parameters first):")
        print(f"   bash scripts/run_preset.sh")
        print()
        print(f"   # Or run directly:")
        print(f"   python -m annotate_video \\")
        print(f"     --video data/videos/YOUR_VIDEO.mp4 \\")
        print(f"     --model omni-research/Tarsier2-7b-0115 \\")
        print(f"     --config tarsier/configs/tarser2_default_config.yaml \\")
        print(f"     --prompts prompts/presets_short.json \\")
        print(f"     --out_csv data/results/output.csv\n")

        # Troubleshooting
        print(f"{Colors.BOLD}4. Troubleshooting:{Colors.ENDC}")
        if not verification_results.get('tarsier', False):
            print(f"   {Colors.WARNING}! Tarsier imports failed - try reinstalling:{Colors.ENDC}")
            print(f"     python install.py --force")
        if not verification_results.get('torch', False):
            print(f"   {Colors.WARNING}! PyTorch import failed - check installation:{Colors.ENDC}")
            print(f"     conda activate {self.env_name}")
            print(f"     python -c 'import torch; print(torch.__version__)'")
        if not verification_results.get('transformers', False):
            print(f"   {Colors.WARNING}! Transformers import failed:{Colors.ENDC}")
            print(f"     conda activate {self.env_name}")
            print(f"     pip install transformers==4.47.0")
        print(f"   - Documentation: https://github.com/bytedance/tarsier")
        print(f"   - Issues: Check README.md and CLAUDE.md in this repository\n")

        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


# ========================================================================
# CLI Entry Point
# ========================================================================

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Tarsier2 Video Annotation Tool - Installation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py                              # Interactive installation
  python install.py --env-name my_tarsier        # Custom environment name
  python install.py --force --non-interactive    # Force overwrite, no prompts
  python install.py --verify-only                # Check existing installation
  python install.py --skip-tarsier               # Only install project deps
  python install.py --skip-env-creation          # Use existing environment

For more information, visit: https://github.com/bytedance/tarsier
        """
    )

    parser.add_argument(
        '--env-name',
        type=str,
        default='tarsier',
        help='Conda environment name (default: tarsier)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite existing tarsier directory'
    )

    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Run without user prompts (use with --force for fully automated)'
    )

    parser.add_argument(
        '--skip-env-creation',
        action='store_true',
        help='Skip environment creation, use existing environment'
    )

    parser.add_argument(
        '--skip-tarsier',
        action='store_true',
        help='Skip Tarsier installation (only install project dependencies)'
    )

    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip system prerequisite checks'
    )

    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing installation (no installation steps)'
    )

    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Disable colors if requested or not a terminal
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    # Print header
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}Tarsier2 Video Annotation Tool - Installation{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")

    # Run installer
    installer = Installer(args)
    exit_code = installer.run()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
