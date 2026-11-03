#!/bin/bash
# ==============================================================================
# Brev Launchable Setup Script:
# uv + Python 3.12 + locked environment + CUDA-safe Jupyter kernel
# ==============================================================================

set -euo pipefail

# --- CONFIGURATION ---
WORKSPACE_DIR="/home/ubuntu/workspace"
REPO_URL="https://github.com/bharats/aaai-symposium-hyper-tutorial.git"
REPO_NAME="aaai-symposium-hyper-tutorial"
REPO_DIR="$WORKSPACE_DIR/$REPO_NAME"

# Pin uv for a reproducible provisioning process.
UV_VERSION="0.13.6"
UV_BIN_DIR="/home/ubuntu/.local/bin"
UV="$UV_BIN_DIR/uv"

VENV_DIR="$REPO_DIR/.venv"
PYTHON_VERSION="3.12"

DATA_DIR="$REPO_DIR/data"
CKPT_DIR="$REPO_DIR/model_checkpoints"

# Jupyter kernel configuration
KERNEL_NAME="hyper-tutorial-py312"
KERNEL_DIR="/home/ubuntu/.local/share/jupyter/kernels/$KERNEL_NAME"
KERNEL_LAUNCHER="/home/ubuntu/.local/bin/${KERNEL_NAME}-kernel"

mkdir -p "$WORKSPACE_DIR"

echo ">>> 1. Installing uv..."
if [ ! -x "$UV" ] || ! "$UV" --version | grep -q "uv $UV_VERSION"; then
    mkdir -p "$UV_BIN_DIR"

    wget -qO- "https://astral.sh/uv/$UV_VERSION/install.sh" \
        | env UV_UNMANAGED_INSTALL="$UV_BIN_DIR" sh
fi

echo ">>> Using: $($UV --version)"

echo ">>> 2. Cloning tutorial repository..."
cd "$WORKSPACE_DIR"

if [ ! -d "$REPO_NAME" ]; then
    git clone "$REPO_URL"
fi

cd "$REPO_DIR"

if [ ! -f "pyproject.toml" ] || [ ! -f "uv.lock" ]; then
    echo "ERROR: pyproject.toml and uv.lock must be committed to the repository."
    exit 1
fi

echo ">>> 3. Installing managed Python $PYTHON_VERSION..."
$UV python install "$PYTHON_VERSION"

echo ">>> 4. Creating project-local virtual environment..."
$UV venv --python "$PYTHON_VERSION" .venv

echo ">>> 5. Syncing exact locked dependencies..."
$UV sync --locked

echo ">>> 6. Registering CUDA-safe Jupyter kernel..."

# PyTorch's CUDA wheels include their own matching cuDNN libraries. Put all
# wheel-provided NVIDIA library directories before Brev's system CUDA paths.
SITE_PACKAGES="$VENV_DIR/lib/python3.12/site-packages"

if [ ! -d "$SITE_PACKAGES/nvidia/cudnn/lib" ]; then
    echo "ERROR: Expected CUDA/cuDNN libraries were not installed in the uv environment."
    echo "Check that your uv.lock includes the CUDA-enabled PyTorch dependencies."
    exit 1
fi

CUDA_LIB_PATH="$SITE_PACKAGES/torch/lib"

while IFS= read -r lib_dir; do
    CUDA_LIB_PATH="$CUDA_LIB_PATH:$lib_dir"
done < <(find "$SITE_PACKAGES/nvidia" -type d -name lib | sort)

mkdir -p "$KERNEL_DIR" "$(dirname "$KERNEL_LAUNCHER")"

cat > "$KERNEL_LAUNCHER" <<EOF
#!/usr/bin/env bash

# Ensure PyTorch loads the CUDA/cuDNN libraries locked in this project,
# before the CUDA libraries supplied by the Brev base image.
export LD_LIBRARY_PATH="$CUDA_LIB_PATH:\${LD_LIBRARY_PATH:-}"

exec "$VENV_DIR/bin/python" -m ipykernel_launcher "\$@"
EOF

chmod +x "$KERNEL_LAUNCHER"

cat > "$KERNEL_DIR/kernel.json" <<EOF
{
  "argv": [
    "$KERNEL_LAUNCHER",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python 3.12 (HyPER Tutorial)",
  "language": "python"
}
EOF

echo ">>> 7. Pre-loading Navier–Stokes dataset..."
mkdir -p "$DATA_DIR"

uvx hf download --repo-type dataset scailab/hyper-navier-stokes-64-1000 ns_data_64_1000.h5 --local-dir "$DATA_DIR/"

uvx hf download --repo-type dataset bharats/hyper-navier-stokes-boundary-64-1000 ns_boundary_data_64_1000.h5 --local-dir "$DATA_DIR/"

echo ">>> 8. Pre-downloading model checkpoints..."
mkdir -p "$CKPT_DIR"

uvx hf download --repo-type model scailab/hyper-models unet-400-lr-0.0001-bs-8-epochs-200-1_model_checkpoint_199_final.pt --local-dir "$CKPT_DIR/"

uvx hf download --repo-type model scailab/hyper-models unet-800-lr-0.0001-bs-8-epochs-200-1_model_checkpoint_199_final.pt --local-dir "$CKPT_DIR/"

uvx hf download --repo-type model scailab/hyper-models refiner_unet_800_shuffle_lr_0.0001_ns_64x64_model_checkpoint_199.pt --local-dir "$CKPT_DIR/"

uvx hf download --repo-type model scailab/hyper-models actor-0.3-reinforce-eta-0.0-lr-1e-05_model_checkpoint_29.pt --local-dir "$CKPT_DIR/"

uvx hf download --repo-type model bharats/hyper-models boundary-unet-800-lr-0.0001-bs-8-epochs-200-1_model_checkpoint_199_final.pt --local-dir "$CKPT_DIR/"

uvx hf download --repo-type model bharats/hyper-models boundary-actor-0.3-reinforce-eta-0.0-lr-1e-05_model_checkpoint_best.pt --local-dir "$CKPT_DIR/"

echo ">>> 9. Fixing permissions..."
sudo chown -R ubuntu:ubuntu \
    "$WORKSPACE_DIR" \
    "$UV_BIN_DIR" \
    "$KERNEL_DIR" \
    "$KERNEL_LAUNCHER"

echo ">>> 10. Starting JupyterLab without authentication..."

JUPYTER_LOG="$WORKSPACE_DIR/jupyterlab.log"
JUPYTER_PID_FILE="$WORKSPACE_DIR/jupyterlab.pid"

nohup "$VENV_DIR/bin/jupyter" lab \
    --ServerApp.root_dir="$REPO_DIR" \
    --ServerApp.ip="0.0.0.0" \
    --ServerApp.port=8888 \
    --ServerApp.port_retries=0 \
    --ServerApp.open_browser=False \
    --ServerApp.allow_remote_access=True \
    --IdentityProvider.token="" \
    --PasswordIdentityProvider.hashed_password="" \
    --PasswordIdentityProvider.password_required=False \
    > "$JUPYTER_LOG" 2>&1 &

JUPYTER_PID=$!
echo "$JUPYTER_PID" > "$JUPYTER_PID_FILE"

sleep 2

if ! kill -0 "$JUPYTER_PID" 2>/dev/null; then
    echo "ERROR: JupyterLab did not start. Recent logs:"
    tail -n 50 "$JUPYTER_LOG"
    exit 1
fi

echo ">>> JupyterLab started on port 8888 with no authentication."
echo ">>> Logs: $JUPYTER_LOG"

echo ">>> SETUP COMPLETE!"
echo ">>> In Jupyter, select: Python 3.12 (HyPER Tutorial)"

