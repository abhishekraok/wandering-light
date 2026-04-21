#!/bin/bash
# One-time setup for a Runpod pod (PyTorch 2.8 template, 4090, 50GB network volume at /workspace).
# Run from anywhere on the pod: `bash runpod_bootstrap.sh`
# Idempotent — safe to re-run.

set -euo pipefail

VOLUME=/workspace
REPO_URL=https://github.com/abhishekraok/wandering-light.git
REPO_DIR=$VOLUME/wandering-light
VENV=$VOLUME/.venv

mkdir -p "$VOLUME"
cd "$VOLUME"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

if [ ! -d "$VENV" ]; then
  python3.12 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -e "$REPO_DIR"

# Point caches at the network volume so they survive pod restarts.
mkdir -p "$VOLUME/hf_cache" "$VOLUME/wandb_runs" "$VOLUME/checkpoints"
ln -sfn "$VOLUME/checkpoints" "$REPO_DIR/checkpoints"

# Persist env for future bash sessions on this pod.
PROFILE=$VOLUME/.env.runpod
cat > "$PROFILE" <<EOF
source $VENV/bin/activate
export HF_HOME=$VOLUME/hf_cache
export WANDB_DIR=$VOLUME/wandb_runs
export PYTHONUNBUFFERED=1
cd $REPO_DIR
EOF
grep -qxF "source $PROFILE" ~/.bashrc || echo "source $PROFILE" >> ~/.bashrc

echo
echo "Done. Open a new shell or: source $PROFILE"
echo "Set these as Runpod pod env vars (or export manually):"
echo "  WANDB_API_KEY, HF_TOKEN, OPENAI_API_KEY, GEMINI_API_KEY"
echo "Then: wandb login \$WANDB_API_KEY && huggingface-cli login --token \$HF_TOKEN"
