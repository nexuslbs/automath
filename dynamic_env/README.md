# dynamic_env - environments as data

A generic engine for the dynamic-nodes environment. Node types, node instances
and the axiom set live in JSON specs under `data/dynamic_env/`; `engine.py`
contains no spec-specific knowledge.

* `spec.py` - spec schema, parsing and validation (`load_spec`,
  `spec_from_dict`).
* `engine.py` - the interpreter (node values, conditions, effects), the derived
  action space, the pure `step`, the state vector and the deterministic planner.
* `tests.py` - 13 deterministic checks (stdlib runner + pytest mirror).
* `demo.py` - CLI: one engine loads every spec from `data/`.

```sh
/opt/automath/venv/bin/python -m dynamic_env.tests
/opt/automath/venv/bin/python -m dynamic_env.demo
```

Design: `docs/evolution/DYNAMIC_NODES.md`; laws:
`docs/evolution/STRICT_LAWS.md`.
