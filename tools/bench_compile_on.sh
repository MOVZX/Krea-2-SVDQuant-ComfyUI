#!/usr/bin/env bash
# Per-step speed benchmark, torch.compile ON (TorchCompileModel, inductor).
# The first run per arm pays inductor's compilation - that is expected.
# Needs a running ComfyUI (default http://127.0.0.1:8188).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/multimedia/AI/ComfyUI/.venv/bin/python"

curl -sf -m 3 http://127.0.0.1:8188/system_stats > /dev/null || {
    echo "ComfyUI not reachable at http://127.0.0.1:8188 - start it first." >&2
    exit 1
}

exec "$PY" "$HERE/speed_bench.py" \
    --server http://127.0.0.1:8188 \
    --arms svdq svdq8 \
    --compile on \
    --out /tmp/speed_compile_on.json \
    "$@"
