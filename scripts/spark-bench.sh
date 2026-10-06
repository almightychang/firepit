#!/usr/bin/env bash
# Build each REV on the GB10 pod and run COMMAND against it, in REV order, on one pod.
# usage: scripts/spark-bench.sh [--out DIR] REV [REV...] -- COMMAND...
# COMMAND runs in the pod's bash with REV, SRC, BIN, and FIREPIT set for each REV.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=$root/scripts/spark.env
usage() {
  echo "usage: scripts/spark-bench.sh [--out DIR] REV [REV...] -- COMMAND..." >&2
  exit 2
}
say() { echo "[spark-bench] $*"; }

[[ -f $config ]] || {
  echo "missing $config: copy scripts/spark.env.example and fill in SPARK_KUBECTL," \
    "SPARK_NODE, SPARK_POD, SPARK_IMAGE, SPARK_HOSTPATH" >&2
  exit 2
}
# shellcheck source=spark.env.example
source "$config"
for name in SPARK_KUBECTL SPARK_NODE SPARK_POD SPARK_IMAGE SPARK_HOSTPATH; do
  [[ -n ${!name:-} ]] || { echo "$config does not set $name" >&2; exit 2; }
done
read -ra kubectl <<<"$SPARK_KUBECTL"

out=$root/runs/spark/$(date -u +%Y%m%dT%H%M%SZ)
revs=()
while (($#)); do
  case $1 in
    --out) [[ $# -ge 2 ]] || usage; out=$2; shift 2 ;;
    --) shift; break ;;
    *) revs+=("$1"); shift ;;
  esac
done
((${#revs[@]} > 0 && $# > 0)) || usage
command=$*

shas=()
for rev in "${revs[@]}"; do
  sha=$(git -C "$root" rev-parse --verify --short=7 "$rev^{commit}") || { echo "unknown REV $rev" >&2; exit 2; }
  [[ " ${shas[*]} " != *" $sha "* ]] || { echo "REV $rev repeats $sha" >&2; exit 2; }
  shas+=("$sha")
done
vendor_sha=$(git -C "$root/vendor/cutile-rs" rev-parse HEAD)
[[ -z $(git -C "$root" status --porcelain --untracked-files=no) ]] \
  || say "working tree has uncommitted changes; each REV is built from its commit only"

allocated=$("${kubectl[@]}" describe node "$SPARK_NODE" | awk '/^  nvidia.com\/gpu  /{print $2}' | tail -1) \
  || { echo "cannot describe node $SPARK_NODE" >&2; exit 1; }
if [[ $allocated != 0 ]]; then
  echo "GPU allocation on $SPARK_NODE is ${allocated:-unknown}; pods requesting a GPU there:" >&2
  "${kubectl[@]}" get pods --all-namespaces --field-selector "spec.nodeName=$SPARK_NODE" \
    -o 'custom-columns=NAMESPACE:.metadata.namespace,NAME:.metadata.name,PHASE:.status.phase,DELETING:.metadata.deletionTimestamp,GPU:.spec.containers[*].resources.limits.nvidia\.com/gpu' \
    | awk 'NR == 1 || $5 != "<none>"' >&2
  exit 1
fi
if "${kubectl[@]}" get pod "$SPARK_POD" >/dev/null 2>&1; then
  echo "pod $SPARK_POD already exists; not touching another run" >&2
  exit 1
fi

mkdir -p "$out"
{
  echo "time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "node: $SPARK_NODE"
  echo "pod: $SPARK_POD"
  echo "image: $SPARK_IMAGE"
  for i in "${!revs[@]}"; do
    echo "rev: ${shas[$i]} $(git -C "$root" rev-parse "${revs[$i]}^{commit}") (${revs[$i]})"
  done
  echo "vendor/cutile-rs: $vendor_sha"
  echo "command: $command"
} >|"$out/meta.txt"

created=false
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if $created; then
    say "deleting pod $SPARK_POD"
    "${kubectl[@]}" delete pod "$SPARK_POD" --wait=true --ignore-not-found >/dev/null || status=1
  fi
  say "results in $out"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

pod() { "${kubectl[@]}" exec "$SPARK_POD" -- "$@"; }
pod_stdin() { "${kubectl[@]}" exec -i "$SPARK_POD" -- "$@"; }

say "creating pod $SPARK_POD on $SPARK_NODE"
sed -e "s|@POD@|$SPARK_POD|" -e "s|@NODE@|$SPARK_NODE|" -e "s|@IMAGE@|$SPARK_IMAGE|" \
  -e "s|@HOSTPATH@|$SPARK_HOSTPATH|" "$root/scripts/spark-pod.yaml" | "${kubectl[@]}" create -f - >/dev/null
created=true
"${kubectl[@]}" wait --for=condition=Ready "pod/$SPARK_POD" --timeout=600s >/dev/null
say "running /work/setup.sh"
pod bash /work/setup.sh >|"$out/setup.log" 2>&1 || { say "setup failed, see $out/setup.log"; exit 1; }

say "uploading vendor/cutile-rs $vendor_sha"
vendor=/work/vendor-$vendor_sha
pod bash -c "rm -rf $vendor && mkdir -p $vendor"
tar -C "$root/vendor/cutile-rs" --exclude=./.git --exclude='./target*' -cf - . | pod_stdin tar -xf - -C "$vendor"

failed=0
env_path=/work/cargo/bin:/usr/local/cuda/bin:/usr/sbin:/usr/bin:/bin
for sha in "${shas[@]}"; do
  src=/work/src-$sha
  target=/work/target-$sha
  say "$sha: uploading source"
  pod bash -c "rm -rf $src && mkdir -p $src/vendor && ln -s $vendor $src/vendor/cutile-rs"
  git -C "$root" archive "$sha" | pod_stdin tar -xf - -C "$src"
  # rustup >= 1.28 installs the source's rust-toolchain.toml (or keeps the default when
  # there is none) with no arguments; older rustup installs it on `show active-toolchain`.
  say "$sha: installing toolchain"
  if ! pod env PATH="$env_path" bash -c \
    "cd $src && { rustup toolchain install --no-self-update || rustup show active-toolchain; } && rustc --version" \
    >|"$out/toolchain-$sha.log" 2>&1; then
    echo "$sha toolchain exit=1" | tee -a "$out/status.txt"
    say "$sha: toolchain install failed, see $out/toolchain-$sha.log"
    exit 1
  fi
  say "$sha: building"
  if ! pod env PATH="$env_path" CARGO_TARGET_DIR="$target" bash -c \
    "cd $src && cargo build --release -p firepit --examples && cargo build --release -p firepit-cli" \
    >|"$out/build-$sha.log" 2>&1; then
    echo "$sha build exit=1" | tee -a "$out/status.txt"
    say "$sha: build failed, see $out/build-$sha.log"
    exit 1
  fi
  say "$sha: running"
  set +e
  pod env PATH="$env_path" CARGO_TARGET_DIR="$target" REV="$sha" SRC="$src" \
    BIN="$target/release/examples" FIREPIT="$target/release/firepit" \
    XDG_CACHE_HOME="/work/runs/spark-bench/jit-$sha" bash -c "cd $src && $command" \
    >|"$out/$sha.log" 2>&1
  status=$?
  set -e
  echo "$sha run exit=$status" | tee -a "$out/status.txt"
  ((status == 0)) || failed=1
done
exit "$failed"
