# PLANNER - explicit bounded search over REDUCTION_TABLE (unit 2b, task 4342)

Branch `gen-fitness`, on top of unit-2a's fitness work. **No learned genome, no
torch, no training** - pure standard library. This is the ALTERNATIVE/COMPARISON
flavor to unit-2a's validation-based fitness term.

## 1. Why (mandate, verbatim)

`COMPARISON.md` section 4, next-fix #3:

> "Replace the learned-score `argmax` at inference with an explicit
> planner/value function over `REDUCTION_TABLE` (search), which generalizes by
> construction rather than by selection."

The scoring path this replaces is `EvolutionAgent._score` / `best_action`
(`new_approach/evolution_agents.py`): `q(goal,state,a) + beta * generalising
term`, argmax over `valid(stack)`. The planner decides the next action by
SEARCH over the reduction table, so the same procedure applies to any target
without any parameters being selected for it.

## 2. World and action alphabets

Same `(target, stack)` stack machine as `EvoEnv` / `MinimalEnv`: an action pops
its `arity` top nodes and pushes one built node.

* `domain="evo"`: `evolution.EVO_ACTIONS` = `PushZero`, `PushOne`,
  `MakeChange`, plus one build action per `REDUCTION_TABLE` row
  (`SEMANTIC_OPS`, `evolution.py:169-175`).
* `domain="core"`: `env.BUILD_ACTIONS` = `PushZero`, `PushOne`, `MakeChange`,
  `MakeGroup2/3/4`.

The canonical build word from the empty stack is `sem_plan(target)` for evo
(the unique post-order word over the semantic kinds; the tag is implicit in the
build action) and `env.plan(target)` for core (the tag is built explicitly).

## 3. Exact algorithm (`new_approach/planner.py`)

`Planner.plan(target, initial_stack) -> Optional[List[str]]`:

1. **Goal / budget gate.** If `initial_stack == (target,)` return `[]`. Compute
   `word = canonical_word(target)`. If `len(word) > max_actions`, return `None`
   (explicit step budget).
2. **Empty-stack fast path.** `initial_stack == ()` returns `word`.
3. **Exact prefix fast path.** Simulate `word` from empty; if the current stack
   equals the stack after `word[:i]`, return `word[i:]`. This is the whole
   `core33` / `ext114` validation families (all non-trivial prefixes of the
   canonical word).
4. **Sound subtree prune.** Every node that survives into a successful run must
   be a SUBTREE of `target` (nodes are only ever consumed into larger nodes and
   the final stack is exactly `target`), and building such a subtree uses only
   actions occurring in `word`. If any current stack node is not a canonical
   subtree of `target`, no solution exists -> return `None` immediately.
5. **Bounded best-first search.** Candidates are the distinct action names of
   `word`, in first-appearance order (deterministic). States are stacks
   (frozen/hashable). Priority is `g + h`:
   * `g` = actions taken;
   * `h(stack)` = the number of canonical build RESULT nodes still missing from
     the stack (multiset difference; `required` comes from simulating `word`
     from empty). It is a lower bound on the remaining actions.
   Ties break on insertion order (monotone counter). `g` is capped by
   `depth_limit = min(max_actions, max(1, len(word)))`. Expanded nodes are
   capped by `max_search_nodes`; a state is never re-expanded with a
   non-improving `g` (Dijkstra-style best-`g`). The search returns the first
   goal state's reconstructed word, else `None`.
6. **Safe fallback.** `best_action(key, state)` returns `plan(...)[0]` when a
   plan exists, otherwise `fallback_action(stack)` = the first action of the
   domain whose `arity <= len(stack)` (always legal; `PushZero`/`PushOne` have
   arity 0). If `target is None` (pattern/ExprGoal goal) it uses the same
   fallback. It never crashes and never returns an illegal action.

**Bounded-search constants** (`planner.py`):

| constant | value | meaning |
| --- | --- | --- |
| `DEFAULT_MAX_SEARCH_NODES` | `100000` | expanded nodes per `plan` call |
| `DEFAULT_MAX_ACTIONS` | `200` | returned plan length / step budget |
| `DEFAULT_MAX_STEPS` | `200` | rollout episode cap |

`plan` is deterministic for a fixed `(target, stack)`: fixed candidate tuple,
counter tie-break, no RNG.

## 4. Completeness argument (why search is not a heuristic shortcut)

Every node on the stack of a run ending in `(target,)` is a subtree of
`target`; the action that builds a subtree `s` is the unique action for `s`'s
root `(tag, arity)`, and that action occurs in `word(target)`. Hence restricting
the search to the action names of `word(target)` loses no solution, and the
subtree prune only removes provably unsolvable states. With an unbounded node
budget the search is complete for reachable `(target,)`; the budget bounds the
work on unsolvable states.

## 5. Driver (`new_approach/planner_eval.py`)

Evaluation-only, mirroring `gen_eval.py`'s CLI and the same three held-out sets:

```
/opt/automath/venv/bin/python -m new_approach.planner_eval \
    --config config/evolution_genfitness.json \
    --out /opt/automath/tmp/planner_eval.json
```

It writes `solved/total/mean_actions_solved/mean_reward/wall_secs` per set plus
`per_form` solved flags and the 40 `unseen_forms`. Same reward convention as
`gen_common.rollout_eval`: `(1 if solved else 0) - 0.05 * actions`. No genome is
loaded, no training happens (`episodes=0`, `train_cases_* = 0`).

## 6. Scope and honest gaps

* The planner solves FIXED-state goals (`FixedStateGoal`). `core33` also
  contains 2 `ExprGoal` pattern-prefix cases with no fixed target; for those the
  driver uses the legal fallback and they are honestly unsolved.
* The search is bounded; if `max_search_nodes` is exhausted on an unsolvable
  state, the plan is `None` and the fallback is returned.
* Unit-2a files (`validation_set.py`, `test_gen_fitness.py`,
  `docs/evolution/GEN_FITNESS.md`) are untouched.
