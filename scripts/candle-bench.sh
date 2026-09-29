#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/container.sh"
case "${1:-}" in
  check) args=(check --release) ;;
  run) shift; args=(run --release -- "$@") ;;
  *) echo 'usage: scripts/candle-bench.sh {check|run [K N B]}' >&2; exit 2 ;;
esac
container -e CUDA_COMPUTE_CAP=120 \
  -e LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64 \
  -e CARGO_TARGET_DIR=/work/target-candle \
  -v "$HOME/Repos/candle:$HOME/Repos/candle:ro" \
  -w /work/tools/candle-q8bench "$image" cargo "${args[@]}"
