#!/bin/bash
# One-time setup for a Runpod pod (PyTorch 2.8 template, 4090, 50GB network volume at /workspace).
# Working copy lives on local disk (/root, fast but ephemeral).
# Durable state (checkpoints, HF/wandb caches, uv wheel cache, persistent repo clone, env profile) lives on /workspace.
# Run from the persistent clone: `bash /workspace/wandering-light/runpod_bootstrap.sh`
# Idempotent — safe to re-run, and called automatically by runpod_post_restart.sh when /root is wiped.

set -euo pipefail

VOLUME=/workspace
LOCAL_ROOT=/root
REPO_URL=https://github.com/abhishekraok/wandering-light.git
REPO_DIR=$LOCAL_ROOT/wandering-light
PYTHON_VERSION=3.12.8

# Persist uv's wheel cache on VOLUME so re-syncs after pod recycle skip the ~3GB torch/CUDA download.
# UV_LINK_MODE=copy because cache (VOLUME/MooseFS) and .venv (local overlay) are different filesystems,
# so uv can't hardlink — this silences the fallback warning and makes behavior explicit.
# (uv's python install dir stays default/local — python tarballs are ~50MB and fast to re-fetch.)
export UV_CACHE_DIR=$VOLUME/uv_cache
export UV_LINK_MODE=copy
mkdir -p "$UV_CACHE_DIR" "$VOLUME/hf_cache" "$VOLUME/wandb_runs" "$VOLUME/checkpoints"

# Install uv if missing.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Clone repo onto local disk (or leave existing clone alone). Fast small-file I/O matters here
# — running tests/training from the volume clone is noticeably slower.
mkdir -p "$LOCAL_ROOT"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

# Use a uv-managed Python that matches uv.lock exactly (not the system 3.12.3).
uv python install "$PYTHON_VERSION"

# Install from uv.lock into the project's default .venv on local disk (fast small-file I/O).
# --extra dev includes pytest/black/ruff so you can run the test suite on the pod.
cd "$REPO_DIR"
uv sync --frozen --extra dev --python "$PYTHON_VERSION"

# Symlink durable paths into the repo.
ln -sfn "$VOLUME/checkpoints" "$REPO_DIR/checkpoints"

# Persist env for future bash sessions on this pod (VOLUME-resident so pod recycles keep it).
PROFILE=$VOLUME/.env.runpod
cat > "$PROFILE" <<EOF
export PATH="\$HOME/.local/bin:\$PATH"
export UV_CACHE_DIR=$VOLUME/uv_cache
export UV_LINK_MODE=copy
export HF_HOME=$VOLUME/hf_cache
export WANDB_DIR=$VOLUME/wandb_runs
export PYTHONUNBUFFERED=1
[ -f "$REPO_DIR/.venv/bin/activate" ] && source "$REPO_DIR/.venv/bin/activate"
[ -d "$REPO_DIR" ] && cd "$REPO_DIR"
EOF
grep -qxF "source $PROFILE" ~/.bashrc || echo "source $PROFILE" >> ~/.bashrc

echo
echo "Done. Open a new shell or: source $PROFILE"
echo "Set these as Runpod pod env vars (or export manually):"
echo "  WANDB_API_KEY, HF_TOKEN, GITHUB_TOKEN"
echo "Then: wandb login \$WANDB_API_KEY && huggingface-cli login --token \$HF_TOKEN"
echo
echo "Set the Runpod pod startup command to: bash $VOLUME/wandering-light/runpod_post_restart.sh"
