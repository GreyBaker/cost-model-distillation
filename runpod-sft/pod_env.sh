#!/usr/bin/env bash
# Source on the Pod: one existing, disposable disk location for the whole job.
# No mount name, second filesystem or lifecycle persistence is assumed.
if [[ -z "${POD_DISK:-}" || ! -d "$POD_DISK" ]]; then
  echo 'Set POD_DISK to an existing writable on-Pod directory before sourcing pod_env.sh.' >&2
  return 1 2>/dev/null || exit 1
fi
export POD_DISK
POD_DISK="$(cd -- "$POD_DISK" && pwd -P)"
export SFT_WORK="${SFT_WORK:-$POD_DISK/sft-work}"
export GKD_WORK="${GKD_WORK:-$POD_DISK/gkd-grid-work}"
export XDG_CACHE_HOME="$POD_DISK/.runtime/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TMPDIR="$POD_DISK/.runtime/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p -- "$PIP_CACHE_DIR" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
