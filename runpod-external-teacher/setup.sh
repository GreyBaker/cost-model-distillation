#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source ./pod_env.sh
case "$(pwd -P)/" in
  "${POD_DISK%/}/"*) ;;
  *) echo 'Code/environments must be on POD_DISK.' >&2; exit 1 ;;
esac
case "${1:---with-reference}" in
  --with-reference) reference=1 ;;
  --native-only) reference=0 ;;
  *) echo 'Usage: bash setup.sh [--with-reference|--native-only]' >&2; exit 1 ;;
esac
index="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python3 -m venv .venv-native
.venv-native/bin/python -m pip install --upgrade pip
.venv-native/bin/python -m pip install 'torch==2.8.0' --index-url "$index"
.venv-native/bin/python -m pip install -r requirements-native.txt
.venv-native/bin/python -m pip check
.venv-native/bin/python -m pip freeze > environment.native.installed.txt
if [[ "$reference" == 1 ]]; then
  python3 -m venv .venv-reference
  .venv-reference/bin/python -m pip install --upgrade pip
  .venv-reference/bin/python -m pip install 'torch==2.8.0' --index-url "$index"
  .venv-reference/bin/python -m pip install -r ../runpod-sft/requirements.txt
  .venv-reference/bin/python -m pip check
  .venv-reference/bin/python -m pip freeze > environment.reference.installed.txt
fi
echo 'Setup complete. Activate .venv-native (NOT the old SFT venv). No model weights downloaded or training launched.'
