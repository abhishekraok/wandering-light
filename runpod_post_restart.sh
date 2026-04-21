#!/bin/bash
# Run after every Runpod pod start. Restores $HOME config from /workspace.
# Usage: bash /workspace/wandering-light/runpod_post_restart.sh

set -euo pipefail

VOLUME=/workspace
REPO_DIR=$VOLUME/wandering-light

# Link dotfiles from volume into $HOME (created fresh on each restart).
ln -sfn "$VOLUME/.bash_custom.sh" "$HOME/.bash_custom.sh"

# Ensure ~/.bashrc sources our env on every new shell.
# Safe to re-run: checks before appending.
if ! grep -qxF "source $VOLUME/.env.runpod" "$HOME/.bashrc" 2>/dev/null; then
    {
        echo ""
        echo "source $VOLUME/.env.runpod"
        echo "source \$HOME/.bash_custom.sh"
    } >> "$HOME/.bashrc"
fi

# wandb and huggingface cache their auth under ~/.netrc / ~/.cache — both ephemeral.
# Re-login using env-var secrets (set via Runpod pod env vars).
# Activate venv so wandb/huggingface-cli are on PATH.
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_DIR/.venv/bin/activate"
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || true
fi

# Git push auth: credential helper reads $GITHUB_TOKEN from env at push time.
# No token on disk; relies on the Runpod Secret being present.
if [ -n "${GITHUB_TOKEN:-}" ]; then
    git config --global credential.helper '!f() { echo username=x-access-token; echo password=$GITHUB_TOKEN; }; f'
fi

echo "Pod restored. Open a new shell (or: source $VOLUME/.env.runpod)"
