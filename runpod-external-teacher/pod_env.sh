#!/usr/bin/env bash
# Source, do not execute. One actual existing writable disposable Pod disk.
if [[ -z "${POD_DISK:-}" || ! -d "$POD_DISK" ]]; then
  echo 'Set POD_DISK to an existing writable on-Pod directory first.' >&2
  return 1 2>/dev/null || exit 1
fi
export EXTERNAL_WORK="${EXTERNAL_WORK:-$POD_DISK/external-xsum-work}"
export GKD_WORK="${GKD_WORK:-$POD_DISK/external-xsum-grid}"
export RESTORED_SFT="${RESTORED_SFT:-$POD_DISK/restored-sft}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../runpod-sft/pod_env.sh"
