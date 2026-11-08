#!/bin/bash
# One-time setup: creates .venv inside the MANTIS folder. Run from the mantis/ folder:  bash hpc/setup_env.sh
set -e
cd "$(dirname "$0")/.."
module load python 2>/dev/null || module load miniconda3 2>/dev/null || true   # see `module avail python`
python3 -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10"; print("python", sys.version.split()[0])'
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r hpc/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pytest -q tests
echo "environment ready: $(pwd)/.venv"
