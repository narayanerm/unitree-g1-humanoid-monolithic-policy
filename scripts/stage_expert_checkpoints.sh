#!/usr/bin/env bash
###############################################################################
# Project    : Humanoid Loco-Manipulation with the Unitree G1                 #
# Developer  : Narayane Ribeiro Medeiros                                      #
# Repository : https://github.com/narayanerm/Humanoid-Loco-Manipulation       #
###############################################################################
# Copy the four expert checkpoints AND the walk terminal-state buffer from the
# sibling repos so this folder is self-contained (no cross-path imports at
# runtime).
#
# The walk buffer is required by policy_pick.reset_from_walk_buffer, which
# fires when the pick env is constructed. Without it the FSM builder crashes
# with FileNotFoundError before the first episode step.
#
# Usage:  bash scripts/stage_expert_checkpoints.sh
#
# The script picks the newest model_*.pt under each expert's g1_<name>/ run
# dir. Override any of the SRC_* paths below if a specific checkpoint is
# preferred (e.g. a converged model_600 instead of the raw newest one).

set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
SIBLING=$(cd "$HERE/.." && pwd)
DEST="$HERE/experts"
mkdir -p "$DEST"

find_newest() {
    # $1: expert folder name (e.g. unitree-g1-humanoid-walk)
    # $2: log experiment (e.g. g1_walk)
    local expert_dir="$SIBLING/$1"
    local log_dir="$expert_dir/outputs/logs/$2"
    if [ ! -d "$log_dir" ]; then
        echo "MISSING $log_dir" >&2
        return 1
    fi
    find "$log_dir" -type f -name "model_*.pt" -printf "%T@ %p\n" \
        | sort -n | tail -1 | awk '{print $2}'
}

copy_expert() {
    # $1: local filename in experts/ (walk.pt/pick.pt/carry.pt/place.pt)
    # $2: source expert folder
    # $3: log experiment name
    local out="$DEST/$1"
    local src
    src=$(find_newest "$2" "$3")
    if [ -z "${src:-}" ] || [ ! -f "$src" ]; then
        echo "!! no checkpoint found for $1 (looked under $2/outputs/logs/$3/)" >&2
        return 1
    fi
    cp -v "$src" "$out"
}

copy_expert walk.pt  unitree-g1-humanoid-walk  g1_walk
copy_expert pick.pt  unitree-g1-humanoid-pick  g1_pick
copy_expert carry.pt unitree-g1-humanoid-carry g1_carry
copy_expert place.pt unitree-g1-humanoid-place g1_drop

# Stage the walk terminal-state buffer used by policy_pick's reset event.
# Any sibling that has run the pick pipeline will have a copy; walk repo is
# the canonical source since it produced the buffer via collect_buffer.
WALK_BUFFER_SRC="$SIBLING/unitree-g1-humanoid-walk/outputs/walk_terminal_states.npz"
WALK_BUFFER_DST="$HERE/outputs/walk_terminal_states.npz"
if [ -f "$WALK_BUFFER_SRC" ]; then
    mkdir -p "$HERE/outputs"
    cp -v "$WALK_BUFFER_SRC" "$WALK_BUFFER_DST"
else
    echo "!! walk_terminal_states.npz not found at $WALK_BUFFER_SRC" >&2
    echo "   Regenerate in the walk repo via: python -m src.collect_buffer --policy-module src.policy_walk --checkpoint <walk_ckpt> --out outputs/walk_terminal_states.npz" >&2
fi

echo "[stage] done. staged assets:"
ls -lh "$DEST"
ls -lh "$HERE/outputs/walk_terminal_states.npz" 2>/dev/null || true
