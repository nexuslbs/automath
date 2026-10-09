# Dynamic-nodes environment (Unit A)

Branch `dynamic-nodes`. Module `dynamic_env/`, spec fixtures `data/dynamic_env/`.

This document specifies the NEW environment that the later evolutionary units
train on. It replaces the flat, fully hardcoded node vocabulary of
`new_approach/` (the four classes `Zero/One/Change/Group` wired directly in
`nodes.py` + `axioms.py`) with a GENERIC engine whose node types, node instances
and axiom set are DATA. The predecessor verdict on thread 4327 (three
generalization approaches G1/G2/G3, `gen-approaches`) is the design input: **no
approach generalised, and a flat per-target preference-vector genome is the
ceiling**; the recommended fix is a target-conditioned / compositional policy.
The environment therefore exposes the TARGET objective to the agent from the
first observation (see section 6), so target conditioning is possible from the
start rather than bolted on later.

---

## 1. Environments are DATA; the engine is generic

The engine (`dynamic_env/engine.py`, `dynamic_env/spec.py`) contains **no spec
id, no node-type name, no tag value and no node-instance data**. Its only
knowledge is:

* how to read a node's value from its declared type semantics;
* a fixed PRIMITIVE instruction set (arithmetic/comparison/logic/sequence ops)
  that a spec SELECTS from, never extends;
* how to interpret a condition tree and an effect list;
* how to derive the action space, step, fire dynamic axioms and plan.

Everything else is a JSON spec file:

| spec element | meaning |
| --- | --- |
| `node_types` | the node-type SCHEMA: `semantics` (`literal` or `combination`), `role` (`value`/`tag`/`objective`/`plain`), `value_field`, and the field list (`int`/`str`/`bool`/`node`/`nodes`/`any`) |
| `nodes` | the node INSTANCES (id -> declared type + field values) |
| `combine_axioms` | rules that give a combination node its value, keyed on `(tag value, arity)` and an `op` primitive |
| `build_axioms` | rule ACTIONS: apply a tag node to operands, constructing a combination node |
| `dynamic_axioms` | axioms whose firing mutates nodes and the rule set (section 3) |
| `objectives` + `guards` | the target assignment and the per-objective set conditions (section 4) |
| `dynamic_node_type`, `max_combine_arity` | the dynamic grouping node action (section 2) |
| `step_cost`, `goal_reward`, `max_steps` | the per-spec reward hook and horizon |

Proof that nothing is hardcoded is executed, not asserted: the test
`no_spec_ids_in_engine` scans `engine.py`/`spec.py` for every shipped spec id
(zero hits), and `adhoc_spec` builds a brand-new inline spec (new type names
`cell/mark/box/goalbit`, new tag `twice`) and solves it with the SAME engine.
The CLI demo loads ALL shipped specs through the same code path.

## 2. Node types, the dynamic grouping node, and the action space

A node is `(id, type, fields)`. Its value is intrinsic only for `literal`
types (the declared `value_field`); a `combination` node has **no intrinsic
meaning**: its value is whatever the ACTIVE combine axiom keyed on
`(tag value, arity)` computes from its operand values. This is the dynamic
grouping node: `tag` is a node id and `items` is a tuple of node ids, so a
combination node expresses a COMBINATION of existing nodes and is the single
construct from which every math/logic expression is defined.

The action space is DERIVED per spec (never declared by hand):

1. `build:<axiom>:<operand>+...` - apply a build axiom (the "apply an
   axiom/rule" action), consuming `arity` evaluable operands and pushing one
   combination node built from the axiom's declared tag node.
2. `combine:<tag-node>:<operand>+...` - combine arbitrary existing evaluable
   nodes under an arbitrary existing TAG node into the dynamic grouping node.
3. `set:<objective>` / `clear:<objective>` - set or clear one elementar 1/0
   objective node (subject to the spec-declared guard).

Order is preserved in operand selection (permutations, not combinations),
because the primitives may be non-commutative (`sub`). `legal_actions` returns
a canonically sorted tuple, so the action space is a deterministic function of
the state. An illegal action is DISCARDED by `try_step` (the state is
unchanged) and raises `IllegalAction` from `step`.

## 3. Axioms define the rules and may be DYNAMIC

Three axiom kinds share one namespace of ids:

* **combine** axioms give combination nodes meaning. A combine axiom is
  `(tag, arity, op, params, active)`. It is consulted only while `active`.
* **build** axioms are the rule-application actions (`tag_node`, `arity`,
  `node_type`, `operand_type`, `active`).
* **dynamic** axioms change the environment DURING an episode. Each is
  `(when, effect, active, once)`. After every action the engine performs ONE
  ordered pass over the active dynamic axioms (sorted by id) and fires each
  whose `when` condition holds on the CURRENT state, then records it as fired
  (default `once`). Effects are deterministic primitives:

  `activate_axiom`, `deactivate_axiom`, `add_node`, `remove_node`,
  `rewrite_node`, `set_objective`.

`spec_dynamic_axiom.json` exercises this: setting the key objective fires
`dyn_unlock`, which ACTIVATES the `build_sub` rule and ADDS node `n7`; the
`build_sub` rule did not exist as a legal action before that state. Because the
firing set is a pure function of the current state (active ids, fired set,
objective flags, node store) and the pass order is fixed, the dynamic axiom set
is still a strict deterministic function of the state; see
`STRICT_LAWS.md` law L5.

## 4. The objective state decomposes into elementar 1/0 objective nodes

The OBJECTIVE STATE is decomposed into elementar objective nodes: declared
nodes whose type has `role: "objective"` and a literal 1/0 value. A spec
declares a TARGET assignment `objectives` (id -> 0/1); the initial value of each
objective node is its declared literal field. Reaching the goal means **every
elementar objective node equals its target** (`goal_reached` is exactly that
conjunction). Objective nodes are never operands, so they cannot be confused
with the working expression store.

An objective may carry a `guard` condition: `set:<objective>` is legal only
while the guard holds. The guards used by the fixtures mention only generic
conditions (`exists_value` over combination nodes, `objective_is`,
`axiom_active`, `and/or/not`), so the guard is DATA too. This is how "build the
expression first, then satisfy the objective" is expressed without hardcoding
which expression is right.

## 5. Shipped spec fixtures and their deterministic single-solution tests

| fixture | what it exercises | min steps | minimal solutions |
| --- | --- | --- | --- |
| `spec_minimal.json` | one elementar objective, no combinations | 1 | 1 |
| `spec_dynamic_group.json` | the dynamic grouping node (`combine`), objective decomposable into an elementar 1/0 node with a combination guard | 2 | 1 |
| `spec_dynamic_axiom.json` | dynamic axioms change the rule set + node store mid-episode | 3 | 1 |
| `spec_multi_step.json` | goal reachable by several paths; min-steps well defined | 2 | 1 (minimal); longer paths exist |

Every spec ships a deterministic single-solution test: `minimal_length` (BFS)
gives min-steps, and `count_goal_words` exhaustively enumerates the legal words
of exactly that length and proves there is EXACTLY ONE minimal action sequence;
`count_goal_words` at every shorter length proves no shorter solution. The
multi-step spec additionally proves a longer correct path exists (length
min+1). A wrong step is either illegal (discarded by `try_step`, state
unchanged, deterministic) or legal and non-goal (it moves the state by the pure
transition law). A reachable goal exists in every shipped spec: the test
`goal_reachable` replays the minimal word and asserts the goal.

## 6. The interface Unit B trains on (target conditioning)

`dynamic_env.engine.DynamicEnv` is the env API:

```python
env = DynamicEnv(load_spec("data/dynamic_env/spec_multi_step.json"))
state = env.reset()                       # State
actions = env.legal_actions(state)        # Tuple[Action, ...]
result = env.step(state, action)          # StepResult(state, reward, done, info)
result, reason = env.try_step(state, a)   # illegal -> (None, reason), state kept
obs = env.observation(state)
vec = env.state_vector(state)             # fixed-length, includes objective + target
```

`state_vector(state)` is `objective flags ++ target flags ++ node-type counts ++
(node count, step, structural fingerprint)`. It therefore ALWAYS exposes the
elementar 1/0 objective nodes AND the target assignment, which is the
target-conditioning seam the predecessor verdict asked for. `observation`
additionally returns the objectives/target as a dict. `Action.key()` is the
stable string the trainer logs. `step` returns the reward hook: `goal_reward`
on the solving step, otherwise `-step_cost`; the total is an exact deterministic
function of the action word (test `reward_hook`).

## 7. Commands

```sh
# deterministic tests (repo venv, Python 3.14)
/opt/automath/venv/bin/python -m dynamic_env.tests
/opt/automath/venv/bin/python -m pytest dynamic_env/tests.py
# smoke demo: the same engine loads every spec from data/
/opt/automath/venv/bin/python -m dynamic_env.demo
```

No new dependency is added: the module is standard-library only.
