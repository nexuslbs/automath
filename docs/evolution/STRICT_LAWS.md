# Strict laws of the dynamic-nodes environment

Branch `dynamic-nodes`, module `dynamic_env`. These are the OPERATOR-BINDING
laws the STATE itself follows. They are not conventions the trainer may opt out
of: the environment is constructed from a spec and has no stochastic transition,
no trainer callback and no external dynamics. Every law below is either
mechanically asserted by `dynamic_env/tests.py` or is a definitional consequence
of the interpreter; the assertion that establishes each one is named.

Notation. A spec `S = (T, N0, A, O*, G, ...)` fixes node types `T`, the initial
node store `N0`, the axiom set `A`, the target objective assignment `O*` and the
objective guards `G`. A state is
`s = (nodes, obj, active, fired, step, next_id, history)`. `s0 = reset(S)`.
`eval_S(s, n)` is the value of node `n`. A legal action is `a ∈ Legal_S(s)`.
The one-step transition is `δ_S(s, a)`; `δ*_S(s, w)` is its extension to an
action word `w`. `goal_S(s)` is the goal predicate. `legal_actions` returns a
canonically sorted tuple and `state.identity()` is the canonical key of the
logical state.

---

## L1 - Deterministic transition

**Statement.** `δ_S` is a total function on legal pairs; for all states `s`,
`s'` and legal actions `a`, `a'`:
`s = s'` and `a = a'` implies `δ_S(s,a) = δ_S(s',a')` as byte-identical states
(`State.canonical()`), and therefore `state_vector(δ_S(s,a)) =
state_vector(δ_S(s',a'))`.

**Proof sketch.** `step` performs a fixed sequence of pure operations:
(A) construct the new node (or set/clear the objective) using only `s` and the
spec; (B) `fire_dynamic_axioms`, which iterates the dynamic axioms sorted by id
and, for each, evaluates a pure condition and applies pure effects; (C) build
the next `State`. A fresh node id is `"g%d" % next_id`, and `next_id` is a
function of `s`; `nodes` and `objectives` are re-sorted by id/key. None of these
reads the clock, `random`, the filesystem, the process environment or any global
mutable state. Hence the result depends only on `(S, s, a)`.

*Asserted by* `check_deterministic_transition`: for every shipped spec the same
`(s, a)` is applied twice and the full canonical states and state vectors must be
equal, and a full minimal word is replayed twice to identical final states.

## L2 - Intrinsic dynamics (no trainer-injected dynamics)

**Statement.** The only transformations applied to the state are the
interpreter primitives declared by the spec's axioms. There is no hook by which
a trainer, reward shaper or environment wrapper can mutate `nodes`, `obj`,
`active` or `fired` during an episode.

**Proof sketch.** `State` is a frozen dataclass; every transition builds a new
value. `DynamicEnv` exposes `step`/`try_step`/`rollout` and read-only queries
(`legal_actions`, `observation`, `state_vector`, `goal_reached`); none accepts a
callback. `fire_dynamic_axioms` applies only the six declared effect primitives
(`activate_axiom`, `deactivate_axiom`, `add_node`, `remove_node`,
`rewrite_node`, `set_objective`). The closed state-evolution law is therefore
the spec itself.

## L3 - Compositional / principled semantics

**Statement.** The value of a node is determined by its subtree and the SPEC,
not by its construction history or its position in the store. Formally, `eval_S`
is the unique algebra homomorphism from the node algebra to the value domain:

* `eval_S(s, n)` is the declared literal for a `literal` node;
* `eval_S(s, n) = op_ax(eval_S(s, tag(n)), [eval_S(s, i) for i in items(n)])`
  for a `combination` node whose tag value and arity select the active combine
  axiom `ax = (tag, arity, op, params)`.

**Proof sketch.** Structural induction on the node tree. A literal node has no
children and is fixed by its field. A combination node's children are `tag` and
the `items`; by the induction hypothesis each child has a unique value; the
active axiom is selected by `(value(tag), len(items))`, a function of the child
values and `active`; `op`/`params` are spec data. Memoisation in `evaluate`
does not change the result because the recursion is well-founded. Two nodes that
are structurally identical (same type, same resolved fields) therefore evaluate
identically regardless of when they were built.

*Corollary.* A grouping node carries no intrinsic meaning: it is only a
combination of existing nodes whose meaning is supplied by the axiom keyed on
its tag and arity. Changing the active axiom set (L5) changes the meaning of the
SAME structure, which is exactly the desired dynamics, and it does so
deterministically.

## L4 - Objective decomposition

**Statement.** The objective state is a vector of elementar 1/0 objective nodes.
For a target assignment `O*` and state `s`,
`goal_S(s) ⇔ ∀ o ∈ dom(O*) : obj_s(o) = O*(o)`.

**Proof sketch.** By definition in `goal_reached`: it iterates the spec's target
ids and requires equality for each; there is no other clause. The initial value
of `obj_s0(o)` is the objective node's declared literal field, and the only
writes are the `set`/`clear` actions and the `set_objective` effect. A goal
exists in every shipped spec (L7).

*Asserted by* `check_objective_decomposition`: `state_vector` starts with the
objective flags and then the target flags, and `goal_reached` equals the manual
conjunction.

## L5 - Dynamic axioms are a deterministic function of the state

**Statement.** Let the active dynamic axioms be `D(s) = {a ∈ A_dyn : a.id ∈
s.active}`, ordered by id. Define `Fire(s)` by threading `s` through
`D(s)`: an axiom `a` is applied iff `a` is active, not already fired when
`a.once`, and `when_a` holds on the state at the moment it is visited. Then
`Fire` is a function of `s` alone, and so is `δ'_S(s,a) = Fire(δ_S(s,a))`.

**Proof sketch.** The order is the fixed total order on ids. Each step of the
fold reads only the threaded state, evaluates a pure condition and applies pure
effects; nothing depends on wall-clock time, iteration order of a hash map or
randomness. `once` membership is read from `s.fired` and updated by the fold.
An axiom activated by an earlier effect but ordered before it simply does not
appear in `D` at its visit and fires on the NEXT step, which is a fixed rule,
not a race.

*Asserted by* `check_dynamic_axiom_fires`: the firing activates `build_sub` (not
legal before), adds exactly `n7`, records `dyn_unlock` in `fired`, and a replay
of the same state gives the byte-identical result.

## L6 - Reward law

**Statement.** For a transition `s --a--> s'` with result `r`:
`r = goal_reward` if `goal_S(s')`, else `r = -step_cost`. For an action word
`w = (a1..ak)` that reaches the goal on the last action,
`Σ r_i = goal_reward - (k-1)·step_cost`; the total is a deterministic function
of `w`.

**Proof sketch.** Immediate from `step`: `done = goal_S(s')`; the branch assigns
`goal_reward` or `-step_cost`. The sum telescopes because the first `k-1` steps
are non-goal. The goal predicate is monotone within a fixed spec? No - a `clear`
action can unset an objective - so the law is stated per word, which is the
honest form: the total is a function of the WORD, hence of the deterministic
trajectory.

*Asserted by* `check_reward_hook`: for every shipped spec the replayed minimal
word's summed reward equals `goal_reward - (k-1)·step_cost` to 1e-9.

## L7 - Reachability

**Statement.** Every shipped spec has a reachable goal: there exists a finite
action word `w` with `goal_S(δ*_S(s0, w))`.

**Proof sketch (constructive).** `minimal_length` is a BFS over the finite
reachable state graph (the action space is finite per state and node ids are
bounded by `next_id`); when it returns a depth, the BFS tree contains a goal
word; `bfs_minimal_word` returns it. `check_goal_reachable` replays that word
through `step` and asserts `goal_reached` on the final state. If the graph search
returned `None` the check fails.

*Asserted by* `check_goal_reachable` (all specs) and `check_minimal_solution_unique`
(exact min-steps).

## L8 - The action space is a deterministic function of the state

**Statement.** `Legal_S(s)` depends only on `(S, s)` and is returned in a fixed
canonical order; `a ∈ Legal_S(s)` is decidable, `step` rejects `a ∉ Legal_S(s)`
with `IllegalAction`, and `try_step` DISCARDS it, returning `(None, reason)`
with the state unchanged.

**Proof sketch.** `legal_actions` enumerates build axioms active in `s` against
permutations of the evaluable operand ids, the combine action against tag nodes
and permutations, and `set`/`clear` per objective subject to the guard; all
inputs are fields of `s` and spec data. The result is sorted by `Action.key()`
and de-duplicated. `is_legal` recomputes the same set. Hence a wrong step is a
deterministic no-op on the state.

*Asserted by* `check_wrong_step_discarded`.

---

## Boundary and honest limits

* The engine's PRIMITIVE set (`sum`, `sub`, `mul`, `if`, `head`, ...) is fixed
  code, not spec data; specs SELECT and COMPOSE primitives but cannot add new
  ones. This is the interpreter boundary (analogous to the four core node types
  of the earlier `new` branch, and stated rather than hidden). Node types, node
  instances, tags, axiom wiring, guards, objectives, reward and horizon are all
  DATA.
* Dynamic axioms fire at most once per episode by default (`once: true`) and one
  ordered pass per step. An axiom that should re-arm can set `once: false`; this
  is not used by the shipped fixtures.
* There is no stochastic transition and no partial observability; `observation`
  exposes the full objective + target. Randomness, if a trainer needs it,
  belongs to the trainer, never to the environment law (L2).
* `state_vector` includes a structural fingerprint (a deterministic rolling hash
  of `state.identity()`); it is an observation feature, not a source of
  transition randomness.
