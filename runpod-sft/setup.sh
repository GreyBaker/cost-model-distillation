#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Run explicitly on the Linux Pod, never as part of plan/tests.
source ./pod_env.sh
case "$(pwd -P)/" in
  "${POD_DISK%/}/"*) ;;
  *) echo 'Place runpod-sft/ on POD_DISK; its .venv must use the same disposable disk.' >&2; exit 1 ;;
esac
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
.venv/bin/python -m pip install 'torch==2.8.0' --index-url "$TORCH_INDEX_URL"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > environment.installed.txt
echo 'Setup finished. Source pod_env.sh and activate .venv in each shell. All job storage is disposable on-Pod disk.'
