#!/bin/bash
# Run after every Runpod pod start. Re-bootstraps the local-disk working copy if it was wiped,
# restores ~/.bashrc sourcing, and re-logs into wandb/huggingface.
# Set the Runpod pod startup command to: bash /workspace/runpod_post_restart.sh
# (This script lives on $VOLUME so it survives pod recycles.)

set -euo pipefail

VOLUME=/workspace
LOCAL_ROOT=/root
REPO_DIR=$LOCAL_ROOT/wandering-light

# Restore Claude Code install + data from volume. Everything under /root is ephemeral
# on Runpod; persisted copies live in $VOLUME/claude/.
CLAUDE_VOL=$VOLUME/claude
if [ -d "$CLAUDE_VOL" ]; then
    mkdir -p "$LOCAL_ROOT/.local/bin" "$LOCAL_ROOT/.local/share" "$LOCAL_ROOT/.local/state"
    ln -sfn "$CLAUDE_VOL/share"       "$LOCAL_ROOT/.local/share/claude"
    ln -sfn "$CLAUDE_VOL/state"       "$LOCAL_ROOT/.local/state/claude"
    ln -sfn "$CLAUDE_VOL/home"        "$LOCAL_ROOT/.claude"
    ln -sfn "$CLAUDE_VOL/config.json" "$LOCAL_ROOT/.claude.json"
    # Point `claude` at the newest installed version (self-updates drop new versions into share/versions).
    LATEST=$(ls "$CLAUDE_VOL/share/versions" 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST" ]; then
        ln -sfn "$CLAUDE_VOL/share/versions/$LATEST" "$LOCAL_ROOT/.local/bin/claude"
    fi
fi

# If local disk was wiped on recycle, re-run bootstrap. Bootstrap is idempotent, so calling it
# when everything's already in place is a fast no-op (uv sync --frozen returns quickly).
if [ ! -d "$REPO_DIR/.git" ] || [ ! -f "$REPO_DIR/.venv/bin/activate" ]; then
    bash "$VOLUME/runpod_bootstrap.sh"
fi

# Link optional custom shell config from volume into $HOME (created fresh on each restart).
if [ -f "$VOLUME/.bash_custom.sh" ]; then
    ln -sfn "$VOLUME/.bash_custom.sh" "$HOME/.bash_custom.sh"
fi

# Ensure ~/.bashrc sources our env on every new shell. Safe to re-run: checks before appending.
if ! grep -qxF "source $VOLUME/.env.runpod" "$HOME/.bashrc" 2>/dev/null; then
    {
        echo ""
        echo "source $VOLUME/.env.runpod"
        [ -f "$VOLUME/.bash_custom.sh" ] && echo "source \$HOME/.bash_custom.sh"
    } >> "$HOME/.bashrc"
fi

# wandb and huggingface cache their auth under ~/.netrc / ~/.cache — both ephemeral.
# Re-login using env-var secrets (set via Runpod pod env vars).
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
