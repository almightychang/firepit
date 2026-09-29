#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/container.sh"
case "${1:-}" in
  check) shift; args=(check "$@") ;;
  test) shift; args=(test "$@") ;;
  *) echo 'usage: scripts/dev.sh {check|test} [cargo args...]' >&2; exit 2 ;;
esac
library_path=/usr/local/cuda/lib64
[[ ${FIREPIT_CUDA_COMPAT:-0} == 1 ]] && library_path=/usr/local/cuda/compat:$library_path
container -e "LD_LIBRARY_PATH=$library_path" \
  -e CARGO_TARGET_DIR=/work/target-c13 -e XDG_CACHE_HOME=/work/target-c13/cache \
  -w /work "$image" cargo "${args[@]}"
