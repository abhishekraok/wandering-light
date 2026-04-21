#!/bin/bash
# Run after every Runpod pod start. Restores $HOME config from /workspace.
# Usage: bash /workspace/wandering-light/runpod_post_restart.sh

set -euo pipefail

VOLUME=/workspace

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
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || true
fi

echo "Pod restored. Open a new shell (or: source $VOLUME/.env.runpod)"
