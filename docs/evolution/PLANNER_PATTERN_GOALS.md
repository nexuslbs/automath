# PLANNER_PATTERN_GOALS - goal-directed planner over ExprGoal/pattern goals

Branch `planner-pattern-goals`, created from `gen-fitness @ 1b8d24e`
(`git branch -r --contains 1b8d24e` -> `origin/gen-fitness`).
Follow-up to `docs/evolution/PLANNER.md` (task 4342, unit 2b) and to the
unsolvable-handling regression (`/opt/automath/evidence/unsolvable-handling/`).

Pure standard library, no torch, no training, no RNG.

## 1. Why (the defect)

`planner.py::planner_rollout()` planned only FIXED-state goals. For a case whose
goal has no `target` attribute (an `ExprGoal`/pattern goal) it did not plan at
all:

```python
target = getattr(case.env.goal, "target", None)
...
else:
    # Non-fixed (pattern) goal: explicit fallback, legal steps only.
    for _ in range(case.max_steps):
        if case.env.goal_achieved(state):
            break
        name = planner.fallback_action(state.stack)
        ...
```

`fallback_action` returns the first legal action in canonical order. From the
`expr_two/prefix1` start `[1]` that is `PushZero`, so the rollout grows the
stack forever and never reaches a single node with value 2. That is exactly why
the only unsolved `core33` form was `expr_two/prefix1`.

Verified BEFORE (raw, base commit `1b8d24e`):

```
SET core33  solved=32/33 ...
SET ext114  solved=114/114 ...
core33 unsolved: {'expr_two/prefix1': False}
```

## 2. The goal formalism (quoted from the fork)

`new_approach/env.py`:

```python
class ExprGoal:
    """A dynamic/pattern goal: a predicate expression over the current node."""

    def __init__(self, build: Callable[[Node], Node], axioms: AxiomSet,
                 name: str = "pattern") -> None:
        self.build = build
        self.axioms = axioms
        self.name = name

    def achieved(self, state: State) -> bool:
        if len(state.stack) != 1:
            return False
        try:
            expr = self.build(state.stack[0])
            return self.axioms.truthy(expr)
        except Exception:
            return False
```

`new_approach/axioms.py` (the predicates the suite exercises):

```python
def _rule_eq(ev, args):
    return ev(args[0]) == ev(args[1])

def _truthy(v: Value) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v != 0
    if isinstance(v, tuple):
        return len(v) > 0
    return bool(v)
```

`new_approach/expressions.py`:

```python
def eq(a: Node, b: Node) -> Group:
    return app(T_EQ, a, b)      # T_EQ = 5
```

The pattern family is therefore `ExprGoal(lambda cur: eq(cur, nat(k)), AX,
name="value==k")`: achieved iff the single stack node evaluates to integer `k`.

## 3. Algorithm: `Planner.plan_pattern`

New method in `new_approach/planner.py`. The goal predicate is the
ENVIRONMENT'S own `env.goal_achieved(state)`, so the search never inspects
`ExprGoal` internals and supports any goal form that implements `achieved`.

Same procedure shape and explicit bounds as the fixed-state search:

1. `start = tuple(initial_stack)`; if the start already satisfies the goal
   return `[]`.
2. Depth cap `limit = min(max_actions, depth_limit)` (the rollout passes
   `min(planner.max_actions, case.max_steps)`).
3. Deterministic candidate order: `self.order`, the domain's canonical action
   order (`BUILD_ACTIONS` for core, `EVO_ACTIONS` for evo) - no RNG.
4. Best-first heap keyed `(g + h, g, tie_counter, stack)` with a monotone
   `itertools.count()` tie-break; a `came` / `gbest` dedup map keyed by the
   reachable stack.
5. Hard `max_search_nodes` expansion cap; returns `None` when the budget or the
   frontier is exhausted.
6. `_pattern_h(stack)` is an admissible lower bound on the actions still needed
   to reach a single-node stack (`len(stack) == 1` is required by `ExprGoal`):
   one action reduces the stack length by at most `max(arity - 1)`, so
   `ceil((len(stack) - 1) / max(arity - 1))`. It only orders the search; it can
   hide no solution.
7. The plan is reconstructed from `came` and returned; otherwise `None`.

`planner_rollout` enters this path ONLY in the `else` branch
(`getattr(goal, "target", None) is None`). The fixed-state branch is the
identical pre-change text; it never reads `pattern_goals`. After a pattern plan
(or when there is none) the SAME safe legal fallback runs for the remaining
budget, so the rollout never crashes and never takes an illegal action.

**Constants** (unchanged from `PLANNER.md`): `DEFAULT_MAX_SEARCH_NODES =
100000`, `DEFAULT_MAX_ACTIONS = 200`, `DEFAULT_MAX_STEPS = 200`. The pattern
suite additionally uses the fork's own oracle bounds `BFS_ORACLE_MAX_ACTIONS =
8`.

## 4. Fixed-state path unchanged

Structural: `plan_pattern` / `_pattern_h` are additive; `planner_rollout`'s
fixed-state branch body is byte-for-byte the old code, and the new branch is
gated by `target is None`. `Planner.pattern_goals` (default `True`) is consulted
only in the pattern branch.

Empirical (same three held-out sets, `planner_eval.py`; raw logs in
`/opt/automath/evidence/planner-pattern-goals/`):

| run | core33 | ext114 | unseen40 |
| --- | ---: | ---: | ---: |
| BEFORE (`1b8d24e`) | 32/33 | 114/114 | 0/40 |
| AFTER, pattern path DISABLED (`--no-pattern-goals`) | 32/33 | 114/114 | 0/40 |
| AFTER, pattern path ENABLED | **33/33** | 114/114 | 0/40 |

Per-form diff `core33` BEFORE vs AFTER DISABLED: `[]`; `ext114` BEFORE vs AFTER
(off and on): `[]`. Per-form diff `core33` BEFORE vs AFTER ENABLED:
`expr_two/prefix1: False -> True`. No other form moves.

## 5. Pattern suite (`new_approach/pattern_suite.py`)

Families (every solvable case is confirmed independently by the fork's own BFS
oracle `bfs_plan(env, max_actions=8)`; the recorded `bfs_optimal_actions` is
that oracle's word length):

* `expr_two` (`eq(cur, nat(2))`): the FULL existing `expr_two/prefix*` start
  family (includes the historic failure `expr_two/prefix1`) plus two fresh
  non-prefix starts.
* `expr_three` (`eq(cur, nat(3))`): prefix starts + two fresh non-prefix starts.
* `expr_one` (`eq(cur, nat(1))`): prefix starts + two fresh non-prefix starts.
* `expr_99` (`eq(cur, nat(99))`): NEGATIVE CONTROL - the oracle returns `None`
  inside `max_actions=8`; the planner must report it UNSOLVED via the legal
  fallback only, never crash, never illegal.

Raw result (13 cases, `pattern_suite.json` + `pattern_suite.trace.txt`):

```
SUMMARY solved=12/13 wall=0.063s
```

`expr_two/prefix1` raw trace:

```
[CASE] expr_two/prefix1 kind=prefix pattern=value==2
  start=[1] max_steps=3
  bfs_oracle: 1 ['MakeChange']
  searched_plan=['MakeChange']
  actions=['MakeChange'] steps=1 solved=True fallback=False
  trace: [1] -> [C(1)]
```

`expr_99/control` raw trace (negative control):

```
[CASE] expr_99/control kind=negative pattern=value==99
  start=[] max_steps=6
  bfs_oracle: none
  searched_plan=[]
  actions=['PushZero', 'PushZero', 'PushZero', 'PushZero', 'PushZero', 'PushZero'] steps=6 solved=False fallback=True
  trace: [] -> [0] -> [0, 0] -> [0, 0, 0] -> [0, 0, 0, 0] -> [0, 0, 0, 0, 0] -> [0, 0, 0, 0, 0, 0]
```

Per-case step count vs BFS-optimal (all twelve solvable cases match the
BFS-optimal length; several find a different optimal word):

| case | bfs optimal | planner steps |
| --- | ---: | ---: |
| expr_two/prefix1 | 1 | 1 |
| expr_two/prefix2 | 0 | 0 |
| expr_two/fresh1 | 2 | 2 |
| expr_two/fresh2 | 2 | 2 |
| expr_three/prefix1 | 2 | 2 |
| expr_three/prefix2 | 1 | 1 |
| expr_three/prefix3 | 0 | 0 |
| expr_three/fresh1 | 3 | 3 |
| expr_three/fresh2 | 2 | 2 |
| expr_one/prefix1 | 0 | 0 |
| expr_one/fresh1 | 1 | 1 |
| expr_one/fresh2 | 1 | 1 |
| expr_99/control (negative) | n/a (unsolvable) | 6 (fallback) |

Determinism: the JSON output carries no wall-clock values, so two full runs are
byte-identical:

```
6912e85615c6649913c085b7f515b6e0aef567c0cdbe0eed874bb1c1f1793328  pattern_suite_run1.json
6912e85615c6649913c085b7f515b6e0aef567c0cdbe0eed874bb1c1f1793328  pattern_suite_run2.json
cmp run1 run2 -> IDENTICAL
```

The trace text differs only in the per-case `wall=` values (timing is
deliberately kept out of the deterministic JSON).

## 6. Comparison (recorded numbers reused)

| route | core33 | ext114 | pattern suite |
| --- | ---: | ---: | ---: |
| planner-pattern (this branch, pattern ON) | **33/33** | 114/114 | **12/13** (negative control unsolved by design) |
| planner-fixed-state-only (this branch, pattern OFF; = `1b8d24e`) | 32/33 | 114/114 | n/a |
| GF-full learned (gen-fitness `COMPARISON.md`, reused) | 14/33 | 12/114 | n/a |
| selection-mean-rate learned route (task 4344 `FINAL_EVIDENCE.md`, reused) | n/a | n/a | unmasked 0-2/30, target-change heldout 0/30 |

Sources for the reused rows: the dispatch quotes gen-fitness `COMPARISON.md`
and selection-mean-rate `FINAL_EVIDENCE.md`. Neither file is present on
`<automath-host>` at the time of writing (the workstation mirror
`/opt/workspace/tmp/automath/selection-mean-rate/` contains only an empty
`unit-3b/`), so these two rows are quoted as recorded, not re-run. The
unsolvable-handling planner regression
(`/opt/automath/evidence/unsolvable-handling/COMPARISON.md`, section D) records
the same fixed-state baseline `core33 32/33, ext114 114/114, unseen40 0/40`.

## 7. Tests

`new_approach.test_planner` (now 13 checks) and `new_approach.tests` (11 checks)
are green. Five new checks cover the pattern path: solve `expr_two/prefix1`,
negative control unsolved with legal fallback only, determinism across two
planners, pattern mode disabled reproduces the legacy fallback, and the
fixed-state path is unchanged with the mode on and off.

## 8. Honest verdict

The planner now generalizes to pattern/`ExprGoal` goals **by construction**: it
searches over reachable stacks with the environment's own `goal_achieved`
predicate and no goal-specific parameters, and it solves every solvable case in
the suite, including the historic `expr_two/prefix1` failure (32/33 -> 33/33)
with zero regression on `ext114` and `unseen40`.

It is not a general pattern *solver*. The exact failure mode is bounded
exhaustion: for a satisfiable pattern deeper than the value chain the suite
exercises (for example `eq(cur, nat(99))` with a large depth cap), the search
would spend its `max_search_nodes` budget and return `None`, after which the
rollout legally falls back and reports the case UNSOLVED. The negative control
in the suite demonstrates exactly that path (unsolved, no crash, legal steps
only), which is the correct honest outcome, not a masked pass.

Concrete next fix: give the pattern search a predicate-aware or distance-to-
single-node bound (for `eq(cur, nat(k))` the target value is known from
`ExprGoal.build`; a numeric lower bound on the remaining value distance would
let it reach `nat(99)` within budget), and wire the same goal-directed search
into `Planner.best_action` for target-less goals (it currently has no env handle
and still returns the legal fallback, which is fine for the fixed-state agent
interface but is the missing piece for closed-loop pattern control).
