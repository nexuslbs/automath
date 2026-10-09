"""CLI demo: ONE generic engine loads DIFFERENT specs from data/.

    /opt/automath/venv/bin/python -m dynamic_env.demo

For every spec file found under ``data/dynamic_env`` (or the ``--spec-dir`` /
``--spec`` arguments) the demo prints the node-type schema and the axiom set it
discovered, shows the objective decomposition (elementar 1/0 objective nodes +
target), asserts the deterministic transition law (same state + same action =
byte-identical next state) and replays the minimal action sequence to the goal,
printing the reward.

Nothing is hardcoded: the demo differs only by the spec FILES it reads.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence

from .engine import DynamicEnv, bfs_minimal_word
from .spec import list_spec_files, load_spec

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_SPEC_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env")


def _action_from_key(env: DynamicEnv, state, key: str):
    for action in env.legal_actions(state):
        if action.key() == key:
            return action
    raise SystemExit("demo: action %r is not legal" % key)


def _show(env: DynamicEnv, path: str) -> None:
    spec = env.spec
    print("=" * 72)
    print("spec file     : %s" % os.path.relpath(path, _REPO_ROOT))
    print("spec id       : %s" % spec.spec_id)
    print("description   : %s" % spec.description)
    print("node types    : %s" % ", ".join(
        "%s(%s,%s)" % (name, nt.semantics, nt.role)
        for name, nt in sorted(spec.node_types.items())
    ))
    print("axioms        : combine=%d build=%d dynamic=%d" % (
        len(spec.combine_axioms), len(spec.build_axioms), len(spec.dynamic_axioms)))
    print("dynamic node  : %s | max_combine_arity=%d" % (
        spec.dynamic_node_type, spec.max_combine_arity))

    start = env.reset()
    print("objective decomposition")
    print("  objective ids : %s" % ", ".join(sorted(spec.objectives)))
    print("  initial flags : %s" % (env.objective_vector(start),))
    print("  target flags  : %s" % (env.target_vector(),))
    print("  state vector  : %s" % (env.state_vector(start),))

    actions = env.legal_actions(start)
    print("legal actions at reset: %d -> %s" % (len(actions), [a.key() for a in actions]))

    # Deterministic transition: same state + same action twice.
    if actions:
        first = env.step(start, actions[0])
        second = env.step(start, actions[0])
        stable = first.state.canonical() == second.state.canonical()
        print("determinism   : same state + %s -> identical next state: %s" % (
            actions[0].key(), stable))
        if not stable:
            raise SystemExit("demo: deterministic-transition law VIOLATED for %s" % spec.spec_id)

    word = bfs_minimal_word(spec)
    if word is None:
        raise SystemExit("demo: no reachable goal in %s" % spec.spec_id)
    state = start
    total = 0.0
    steps = []
    for key in word:
        action = _action_from_key(env, state, key)
        result = env.step(state, action)
        total += result.reward
        steps.append("%s(%.3f)" % (action.key(), result.reward))
        state = result.state
    print("minimal word  : %s" % (list(word),))
    print("replay        : %s" % " -> ".join(steps))
    print("goal reached  : %s | steps=%d | total reward=%.3f" % (
        env.goal_reached(state), len(word), total))
    if not env.goal_reached(state):
        raise SystemExit("demo: goal NOT reached for %s" % spec.spec_id)


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", default=_DEFAULT_SPEC_DIR)
    parser.add_argument("--spec", action="append", default=None,
                        help="load only these spec file stems (repeatable)")
    args = parser.parse_args(argv)

    if args.spec:
        paths = [os.path.join(args.spec_dir, stem + ".json") for stem in args.spec]
    else:
        paths = list_spec_files(args.spec_dir)
    if not paths:
        raise SystemExit("demo: no spec files under %s" % args.spec_dir)

    print("ONE generic engine, %d DIFFERENT specs from data (no engine change)" % len(paths))
    ids: List[str] = []
    for path in paths:
        env = DynamicEnv(load_spec(path))
        ids.append(env.spec.spec_id)
        _show(env, path)
    print("=" * 72)
    print("SMOKE OK: same engine loaded %d different specs: %s" % (len(ids), ", ".join(ids)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
