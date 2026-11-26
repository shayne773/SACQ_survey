# Source me. pi0.5 SACQ lab environment.
# LAB is this experiment folder; outputs (sacq/, runs/, reset/, verify/, logs/) are written here (git-ignored).
# Required: CKPT (openpi's PyTorch conversion of pi05_libero). Optional: OPENPI_DIR (default $LAB/openpi),
# SACQ_REPO (the SACQ_survey checkout providing the `sacq` package; default: the repo containing this folder).
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPENPI_DIR="${OPENPI_DIR:-$LAB/openpi}"
export SACQ_REPO="${SACQ_REPO:-$(cd "$LAB/../.." && pwd)}"
source "$LAB/toolchain_env.sh"
export LIBERO_CONFIG_PATH=$LAB/libero_config
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl GALLIUM_DRIVER=llvmpipe
export PYTHONPATH=$OPENPI_DIR/third_party/libero:$LAB/scripts:$SACQ_REPO
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PY=$OPENPI_DIR/.venv/bin/python
export CKPT="${CKPT:?set CKPT to the pi05_libero PyTorch checkpoint directory (see README)}"
export TORCHINDUCTOR_COMPILE_THREADS=1 TORCHDYNAMO_DISABLE=1
