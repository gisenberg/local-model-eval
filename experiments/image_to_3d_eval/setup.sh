#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="$ROOT/third_party"
HF_HOME_DEFAULT="$ROOT/hf_cache"

TRELLIS_REPO="https://github.com/microsoft/TRELLIS.2.git"
TRELLIS_COMMIT="75fbf0183001ed9876c8dbb35de6b68552ee08bd"
HUNYUAN_REPO="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
HUNYUAN_COMMIT="82920d643c0dc2f7bfd7255f45f62d386edfe60c"

usage() {
  cat <<'EOF'
Usage: ./setup.sh [--check] [--clone] [--hunyuan] [--trellis] [--all]

  --check     Print local GPU/tooling status.
  --clone     Clone/update pinned upstream repositories only.
  --hunyuan   Clone repos and install Hunyuan3D 2.1 into .venv-hunyuan with uv.
  --trellis   Clone repos and run TRELLIS.2's upstream Conda installer.
  --all       Run --check, --clone, --hunyuan, and --trellis.
EOF
}

check_env() {
  echo "== GPU =="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi not found"
  fi

  echo
  echo "== Tools =="
  for tool in git uv python3 conda; do
    if command -v "$tool" >/dev/null 2>&1; then
      printf '%-8s %s\n' "$tool" "$(command -v "$tool")"
    else
      printf '%-8s missing\n' "$tool"
    fi
  done
  if command -v nvcc >/dev/null 2>&1; then
    printf '%-8s %s\n' "nvcc" "$(command -v nvcc)"
  elif [ -x /usr/local/cuda/bin/nvcc ]; then
    printf '%-8s %s\n' "nvcc" "/usr/local/cuda/bin/nvcc"
  else
    printf '%-8s missing\n' "nvcc"
  fi
}

clone_at_commit() {
  local repo_url="$1"
  local dest="$2"
  local commit="$3"

  mkdir -p "$THIRD_PARTY"
  if [ ! -d "$dest/.git" ]; then
    git clone --recursive "$repo_url" "$dest"
  fi

  git -C "$dest" fetch --tags origin
  git -C "$dest" checkout "$commit"
  git -C "$dest" submodule update --init --recursive
}

clone_repos() {
  clone_at_commit "$TRELLIS_REPO" "$THIRD_PARTY/TRELLIS.2" "$TRELLIS_COMMIT"
  clone_at_commit "$HUNYUAN_REPO" "$THIRD_PARTY/Hunyuan3D-2.1" "$HUNYUAN_COMMIT"
}

install_hunyuan() {
  clone_repos
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required for the Hunyuan environment. Install uv first." >&2
    exit 1
  fi

  local repo="$THIRD_PARTY/Hunyuan3D-2.1"
  local venv="$ROOT/.venv-hunyuan"

  if [ ! -x "$venv/bin/python" ]; then
    uv venv --python 3.10 "$venv"
  fi
  # shellcheck disable=SC1091
  source "$venv/bin/activate"

  export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
  export PIP_EXTRA_INDEX_URL="https://mirrors.cloud.tencent.com/pypi/simple/ https://mirrors.aliyun.com/pypi/simple"

  local compute_cap=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
  fi
  if [[ "$compute_cap" == 12.* ]]; then
    echo "Detected Blackwell compute capability $compute_cap; installing PyTorch CUDA 13 wheels for sm_120 support."
    uv pip install 'torch==2.12.0' 'torchvision==0.27.0'
  else
    uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  fi
  uv pip install setuptools wheel cython numpy==1.24.4
  uv pip install basicsr==1.4.2 --no-build-isolation --no-deps
  # The upstream requirements include Blender's Python package and deepspeed for
  # demo/training paths. The direct inference wrappers here do not need them,
  # and bpy==4.0 is not available for every Python/index combination.
  uv pip install -r <(grep -Ev '^(bpy==|deepspeed$)' "$repo/requirements.txt") --no-build-isolation --index-strategy unsafe-best-match
  if [[ "$compute_cap" == 12.* ]]; then
    # Some indirect dependencies currently prefer stable CUDA 13 torch wheels.
    # Re-pin to cu128 before building CUDA extensions so it matches /usr/local/cuda 12.8.
    uv pip install --pre --force-reinstall 'torch==2.12.0.dev20260408+cu128' --index-url https://download.pytorch.org/whl/nightly/cu128
    uv pip install --pre --force-reinstall --no-deps 'torchvision==0.27.0.dev20260407+cu128' --index-url https://download.pytorch.org/whl/nightly/cu128
  fi

  if ! (cd "$repo/hy3dpaint/custom_rasterizer" && uv pip install -e . --no-build-isolation); then
    echo "Warning: Hunyuan paint custom_rasterizer did not build. Shape generation can still run." >&2
    echo "         Texture generation requires a CUDA toolkit that matches torch.version.cuda." >&2
  fi
  (
    cd "$repo/hy3dpaint/DifferentiableRenderer"
    bash compile_mesh_painter.sh || true
    suffix="$(python - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)"
    if [ ! -f "mesh_inpaint_processor${suffix}" ]; then
      c++ -O3 -Wall -shared -std=c++11 -fPIC \
        $(python -m pybind11 --includes) \
        mesh_inpaint_processor.cpp \
        -o "mesh_inpaint_processor${suffix}"
    fi
  )

  mkdir -p "$repo/hy3dpaint/ckpt"
  if [ ! -f "$repo/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ]; then
    (cd "$repo" && python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve

out = Path("hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
urlretrieve(url, out)
PY
    )
  fi
}

install_trellis() {
  clone_repos
  if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required for TRELLIS.2's upstream installer." >&2
    echo "Install Miniconda/Mambaforge, then rerun: ./setup.sh --trellis" >&2
    exit 1
  fi

  local repo="$THIRD_PARTY/TRELLIS.2"
  if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d /usr/local/cuda ]; then
      export CUDA_HOME=/usr/local/cuda
    elif [ -d /usr/local/cuda-12.4 ]; then
      export CUDA_HOME=/usr/local/cuda-12.4
    fi
  fi
  if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/bin" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
  fi
  (
    cd "$repo"
    # Upstream setup.sh expects to be sourced so it can activate the new env.
    # shellcheck disable=SC1091
    source ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
  )
}

if [ "$#" -eq 0 ]; then
  usage
  exit 0
fi

for arg in "$@"; do
  case "$arg" in
    --check) check_env ;;
    --clone) clone_repos ;;
    --hunyuan) install_hunyuan ;;
    --trellis) install_trellis ;;
    --all) check_env; clone_repos; install_hunyuan; install_trellis ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done
