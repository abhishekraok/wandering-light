#!/bin/bash
# One-time setup for a Runpod pod (PyTorch 2.8 template, 4090, 50GB network volume at /workspace).
# Run from anywhere on the pod: `bash runpod_bootstrap.sh`
# Idempotent — safe to re-run.

set -euo pipefail

VOLUME=/workspace
REPO_URL=https://github.com/abhishekraok/wandering-light.git
REPO_DIR=$VOLUME/wandering-light
PYTHON_VERSION=3.12.8

mkdir -p "$VOLUME"
cd "$VOLUME"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

# Install uv (fast resolver + lockfile-based sync).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Use a uv-managed Python that matches uv.lock exactly (not the system 3.12.3).
uv python install "$PYTHON_VERSION"

# Install from uv.lock into the project's default .venv.
# --extra dev includes pytest/black/ruff so you can run the test suite on the pod.
cd "$REPO_DIR"
uv sync --frozen --extra dev --python "$PYTHON_VERSION"

# Point caches at the network volume so they survive pod restarts.
mkdir -p "$VOLUME/hf_cache" "$VOLUME/wandb_runs" "$VOLUME/checkpoints"
ln -sfn "$VOLUME/checkpoints" "$REPO_DIR/checkpoints"

# Persist env for future bash sessions on this pod.
PROFILE=$VOLUME/.env.runpod
cat > "$PROFILE" <<EOF
export PATH="\$HOME/.local/bin:\$PATH"
source $REPO_DIR/.venv/bin/activate
export HF_HOME=$VOLUME/hf_cache
export WANDB_DIR=$VOLUME/wandb_runs
export PYTHONUNBUFFERED=1
cd $REPO_DIR
EOF
grep -qxF "source $PROFILE" ~/.bashrc || echo "source $PROFILE" >> ~/.bashrc

echo
echo "Done. Open a new shell or: source $PROFILE"
echo "Set these as Runpod pod env vars (or export manually):"
echo "  WANDB_API_KEY, HF_TOKEN, GITHUB_TOKEN"
echo "Then: wandb login \$WANDB_API_KEY && huggingface-cli login --token \$HF_TOKEN"
