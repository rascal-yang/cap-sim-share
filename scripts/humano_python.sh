#!/usr/bin/env bash
set -euo pipefail
CAP_SIM_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CAP_SIM_HUMANO_ROOT="${CAP_SIM_HUMANO_ROOT:-/home/kinetra-4090-01/HUMANO}"
CAP_SIM_TRT_ROOT="${CAP_SIM_TRT_ROOT:-/home/kinetra-4090-01/.local/share/cap-sim/runtime/tensorrt-10.13.0.35}"
export CAP_SIM_HUMANO_ROOT
export LD_LIBRARY_PATH="${CAP_SIM_TRT_ROOT}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${CAP_SIM_REPO_DIR}/src:${CAP_SIM_TRT_ROOT}/usr/lib/python3.10/dist-packages${PYTHONPATH:+:${PYTHONPATH}}"
"${CAP_SIM_HUMANO_ROOT}/.venv/bin/python" "${CAP_SIM_REPO_DIR}/scripts/check_humano_runtime.py" >&2
if [[ "${1:-}" == "--check-only" ]]; then
    exit 0
fi
exec "${CAP_SIM_HUMANO_ROOT}/.venv/bin/python" "$@"
