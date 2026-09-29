root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=firepit-dev:cuda13.3

container() {
  docker image inspect "$image" >/dev/null 2>&1 \
    || docker build -q -t "$image" -f "$root/docker/dev.Dockerfile" "$root/docker" >/dev/null
  docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -e "HOME=$HOME" -e "CARGO_HOME=$HOME/.cargo" -e "RUSTUP_HOME=$HOME/.rustup" \
    -e "PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:/usr/sbin:/usr/bin:/bin" \
    -e CUDA_HOME=/usr/local/cuda -e CUDA_TOOLKIT_PATH=/usr/local/cuda \
    -v "$HOME/.cargo:$HOME/.cargo" -v "$HOME/.rustup:$HOME/.rustup" \
    -v "$HOME/Models:$HOME/Models:ro" \
    -v "$root:/work" "$@"
}
