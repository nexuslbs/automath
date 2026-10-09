# AGENTS.md - automath repository conventions

Durable facts for any worker/agent that touches this repository. Keep this file
short and current; project conventions here win over generic templates.

## Layout

* `new_approach/` - the "4 core node types" approach (`nodes.py`, `axioms.py`,
  `env.py`) plus the evolution pipeline (`evolution_*.py`). Do NOT change the
  core node-type count; new vocabulary is derived.
* `dynamic_env/` - the GENERIC dynamic-nodes environment (branch
  `dynamic-nodes`): node types, node instances and axioms are DATA in
  `data/dynamic_env/*.json`; the engine contains no spec-specific knowledge.
* `env/`, `agent/`, `utils/` - the upstream/reference implementation.
* `docs/evolution/` - design docs and evidence for the evolution work
  (`DESIGN.md`, `REWARDS.md`, `DYNAMIC_NODES.md`, `STRICT_LAWS.md`).

## Test commands

The project venv lives at `/opt/automath/venv` on the automath host. Standard
library test runners are the primary contract; pytest is a mirror.

```sh
/opt/automath/venv/bin/python -m new_approach.tests            # core, 11 checks
/opt/automath/venv/bin/python -m new_approach.evolution_tests  # extended
/opt/automath/venv/bin/python -m dynamic_env.tests             # dynamic env, 13 checks
/opt/automath/venv/bin/python -m pytest dynamic_env/tests.py   # pytest mirror
```

The environment for the dynamic-nodes work is standard-library only; do not add
a dependency to build or test it.

## Dynamic-nodes environment contract

* Build an environment ONLY from a spec: `DynamicEnv(load_spec(path))` or
  `spec_from_dict(raw)`. Environments are data; the engine is generic.
* All state changes go through `step`/`try_step`. The transition is a pure
  function of `(spec, state, action)`; never mutate a `State` (it is frozen) and
  never inject trainer dynamics. See `docs/evolution/STRICT_LAWS.md`.
* The state vector always exposes the elementar 1/0 objective nodes AND the
  target, so a policy can be target-conditioned from the first observation.
* Reward is `goal_reward` on the solving step and `-step_cost` otherwise.
* A new environment is a new JSON spec under `data/dynamic_env/`, never a code
  branch in the engine. The test `no_spec_ids_in_engine` enforces this.

## Branches

`main` is the base. Evolution work lives on `evolution`, `semi-evo`,
`pure-evo`, `gen-approaches` and `dynamic-nodes`. All commits and pushes are
made from the workstation clone with the GitHub App credential helper; never
commit a token, key or host address.
