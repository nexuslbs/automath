#!/usr/bin/env bash
# One-command wrapper: evaluate a list of trained seeds with the standard
# held-out harness and write one JSON (task 4349 unit 5).
#
# usage:
#   run_heldout_eval.sh OUT.json RUN_ROOT SEED [SEED ...]
#
# Examples (the real evaluation the NEXT unit runs):
#   ./run_heldout_eval.sh /tmp/heldout.json \
#       /opt/automath/tmp/unsolvable-unit4-pop 7 13 42
#
#   EPISODES=30 FAMILIES=unseen40,val64,spec_multi_step \
#       ACTION_SET=both ./run_heldout_eval.sh /tmp/heldout.json RUN_ROOT 7
#
# Seed dirs are RUN_ROOT/seed-<SEED>.  Env overrides: PYTHON (interpreter,
# default the project venv), EPISODES, EPISODE_SEED, EPSILON, STEP_BUDGET,
# MAX_WALL_SECS, FAMILIES, ACTION_SET.
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 OUT.json RUN_ROOT SEED [SEED ...]" >&2
    exit 2
fi

OUT="$1"; shift
RUN_ROOT="$1"; shift

PY="${PYTHON:-/opt/automath/venv/bin/python}"

args=(--out "$OUT")
args+=(--episodes "${EPISODES:-30}")
args+=(--episode-seed "${EPISODE_SEED:-20261011}")
args+=(--epsilon "${EPSILON:-0.1}")
args+=(--step-budget "${STEP_BUDGET:-40}")
args+=(--max-wall-secs "${MAX_WALL_SECS:-900}")
args+=(--action-set "${ACTION_SET:-both}")
if [ -n "${FAMILIES:-}" ]; then
    args+=(--families "$FAMILIES")
fi
for seed in "$@"; do
    args+=(--seed-dir "$RUN_ROOT/seed-$seed")
done

echo "+ $PY -m evolution_trainer.heldout_harness ${args[*]}" >&2
exec "$PY" -m evolution_trainer.heldout_harness "${args[@]}"
