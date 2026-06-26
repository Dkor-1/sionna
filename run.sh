#!/usr/bin/env bash
# Convenience launcher for the Sionna passive-radar pipeline.
# `conda activate sionna` is currently BROKEN (the env is installed for
# /workspace/jeong/... but that path does not exist in this container — see
# ENV_NOTES.md).  Work around it by calling the env's python directly.
set -euo pipefail

PY=/home/yunjung/workspace/jeong/miniforge3/envs/sionna/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$(dirname "$0")"
exec "$PY" passive_radar_stage1.py "$@"
