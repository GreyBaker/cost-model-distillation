#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Override explicitly for the Studio driver; never silently install CPU PyTorch.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
.venv/bin/python -m pip install 'torch>=2.6,<3' --index-url "$TORCH_INDEX_URL"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > environment.installed.txt
echo 'Setup finished. Activate .venv, choose persistent TEACHER_WORK, then run teacher.py plan.'
