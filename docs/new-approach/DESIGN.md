# A minimal-node approach for automath (branch `new`)

Unit U1 of the operator orchestrator task: design + implementation + deterministic
single-solution tests for an entirely new minimal-node approach.

## 0. Mechanism choice and branch point

**Choice: a NEW branch `new`** (the primary option in the dispatch). The code is
additionally isolated in a new package directory `new_approach/` so the legacy
`env/`, `agent/`, `test_suite/` trees are untouched, but the branch is the
mechanism that carries the work.

* Branch point: `746022d8d83230949b7b50fe153166e9c86fc5a6` (`main` HEAD).
* Branch: `new`, pushed to `origin`.
* Code: `new_approach/` (see section 7 for the test command).

The legacy suite is NOT run (it OOMs the 3.8 GB no-swap remote box); the new
tests are pure standard library, tiny and bounded.

## 1. Why a new approach

The legacy fork exposes 24 curated primitive actions (`ESSENTIAL_ACTIONS`), a
50-entry node-type catalogue, and states of ~28k tensor rows. The operator's
mandate is to reduce math and logic to the MINIMUM node types, derive the rest,
and add a DYNAMIC node that expresses a COMBINATION of existing nodes so that
every math and logic expression can be defined. This document does that and
measures how far the reduction goes.

## 2. The minimal node set: FOUR types (3 basic + 1 dynamic)

| # | node type | kind | semantics | what it derives |
| - | --------- | ---- | --------- | --------------- |
| 1 | `Zero` | nullary basic | `0`, boolean **false**, empty combination | numbers, booleans, list terminator |
| 2 | `One` | nullary basic | `1`, boolean **true** | the unit of arithmetic and truth |
| 3 | `Change` | unary basic | `Change(x)`: the state/action step, `value(x)+1` | successor, the Action/State transition |
| 4 | `Group` | **dynamic n-ary** | `Group(tag, items)`: a COMBINATION of existing nodes, meaning deferred to the axioms | every operator, sequence, list, control-flow form, and meta definition |

A node's *type* is its class; there are exactly four. `node_type_count` in the
test suite asserts the entire suite uses only these four.

Why these three basic nodes: `Zero`/`One` are the operator's objective facts
(Off/0, On/1); `Change` is the operator's Change/Action applied to a state; the
dynamic `Group` replaces the whole operator vocabulary. `Axiom`, `Goal` and
`State` are NOT node types here (section 4), which is the main departure from
the reference (section 10).

## 3. The dynamic grouping node

`Group(tag, items)` is an n-ary node whose `tag` is itself a node and whose
`items` are operands. It has **no intrinsic meaning**: an `AxiomSet` maps the
pair `(value(tag), len(items))` to a rule. Consequences:

* Every operator is DERIVED, not declared: `ADD` is the tag `One` (value 1),
  `LT` is the tag `Change(Change(Change(One)))` (value 4), and so on. No
  operator adds a node type.
* Arity is dynamic: `Group(tag, ())`, `Group(tag, (a,))`, `Group(tag, (a,b,c))`
  are all the same node type; `SEQ` uses this directly.
* Tags are numerals built only from `Zero`/`One`/`Change`, so the operator
  alphabet is itself derived from the three basic nodes.

The legacy design declares 24 action types and 50 node types; here there is
ONE combination node and a data-driven axiom table.

## 4. The meta layer (subjective understanding over groups of nodes)

Both meta objects are ordinary data structures, not node types:

* **Axiom** `Axiom(tag_value, arity, name, rule)`. An axiom is a rule/law that
  says what a combination of nodes means *in a context*. `rule` receives the
  recursive evaluator and the RAW operand nodes, so `IF` can be lazy (real
  control flow, not eager evaluation of both branches).
* **Goal**. `FixedStateGoal(target)` is the reference's simplest goal: achieved
  iff the state is EXACTLY the fixed target. `ExprGoal(build)` is the dynamic /
  pattern goal: achieved iff the predicate expression built from the current
  node evaluates true under the axioms.

Reference semantics, kept: a **Change applied to a State produces a new State
according to the Axioms**, and the **Goal verifies the new State satisfies it
according to the Axioms**. The state is a stack of nodes; an action consumes the
top `arity` nodes and pushes the constructed node. `Change` is the unary state
transition; `Group` is how combinations (including axiom applications such as
`VerifyGoal`) are built.

## 5. Expressiveness argument (how arbitrary math/logic is encoded)

Canonical forms (all nodes only from the four types; `C(x)` = `Change(x)`,
`G(t;a,b)` = `Group(tag=t, items=(a,b))`):

| concept | encoding | canonical example |
| ------- | -------- | ----------------- |
| number `n` | `nat(n)`: `0,1,C(1),C(C(1)),...` | `3` = `C(C(1))` |
| boolean | `Zero` / `One` | `true` = `1` |
| operator tag | a numeral | `ADD` = `1`, `LT` = `C(C(C(1)))` |
| application `op(a,b)` | `Group(nat(op), (a,b))` | `add(2,3)` = `G(1;C(1),C(C(1)))` |
| comparison | `LT`/`EQ` tags | `lt(2,3)` = `G(C(C(C(1)));C(1),C(C(1)))` |
| sequence | `Group(SEQ, items)` dynamic arity | `[1,2,3]` = `G(C(C(C(C(C(C(C(C(C(1))))))))); 1,C(1),C(C(1)))` |
| control flow | `Group(IF,(c,t,e))`, lazy | `if(lt(1,2),add(1,1),0)` |

Any finite AST is a `Group` with a tag child and operand children, whose leaves
are `Zero`/`One` and whose unary chains are `Change`. Because the tag is itself
any node, arbitrary operators, named functions and nested expressions are
representable. The test suite evaluates 11 samples spanning numbers, booleans,
comparison, arithmetic, logical connectives, lazy control flow, sequences and
list access — all on the 4-node encoding.

Closure of the encoders: `nat`, `app`, and the sample builders only ever emit
`Zero`, `One`, `Change`, `Group`. There is no escape hatch to a fifth type.

## 6. Reduction attempt (measured, not asserted)

The test `reductions` runs the SAME 11 samples through each encoding and asserts
identical results:

| types | basic nodes | construction | demonstrated by |
| ----: | ----------- | ------------ | --------------- |
| 4 | `Zero`, `One`, `Change` | + dynamic `Group` (target) | `to_min4` (native) |
| 3 | `Zero`, `Change` | `One := Change(Zero)` | `to_min3` + `AxiomSet.evaluate` |
| 2 | `Zero` | `Change(x) := Group(Zero,(x,))` | `to_min2` + `AxiomSet.evaluate` |
| 1 | `Cell` | everything is one untyped n-ary container | `to_cell` + `cell_value` |

Measured (remote, section 8): all 11 samples evaluate to the same values under
4, 3, 2 and 1 node types.

### Is `< 3` possible? (operator asked to report if otherwise)

**Yes, `< 3` is possible — under a strict syntactic reading the minimum is 1.**
A single untyped n-ary container `Cell(())` / `Cell((k1,...,kn))` is the
universe of hereditarily finite sets / pure S-expressions; `Zero := Cell(())`,
`One := Cell((Cell(()),))`, `Change` and `Group` are recovered from structure,
and the `cell_value` interpreter evaluates the same axioms. So the operator's
"`< 3` believed impossible" is **not** a consequence of expressiveness; it is
true only if one also requires (a) at least two distinguishable nullary value
nodes (`0` and `1`) and (b) a typed arity distinction between the unary
state-change node and the n-ary combination node. Under that added requirement
the minimum is exactly 4, which the design meets. Both readings are reported
honestly rather than silently choosing one.

### `<= 10` verification table

The operator's 9 "minimum" semantic kinds plus the dynamic grouping node map
onto the four node types as follows — the answer is 4, well inside 10:

| operator slot | semantic need | implemented as | counts as a node type? |
| ---: | ------------- | -------------- | --- |
| 1 | objective Off / 0 | `Zero` | basic |
| 2 | objective On / 1 | `One` | basic |
| 3 | Change / Action | `Change` | basic |
| 4 | boolean false | `Zero` | derived |
| 5 | boolean true | `One` | derived |
| 6 | number / arithmetic operands | `Zero`,`One`,`Change` | derived |
| 7 | comparison / logical operator | `Group` with a numeral tag | derived |
| 8 | sequence / control flow | `Group` (dynamic arity, lazy `IF`) | derived |
| 9 | Goal / Axiom (meta) | meta records over `Group`, not nodes | meta (0) |
| 10 | dynamic grouping | `Group` | dynamic |

## 7. Deterministic single-solution tests

Command (pure standard library; there is no local interpreter, so this is run on
the remote venv):

```sh
cd /opt/automath/repo && /opt/automath/venv/bin/python -m new_approach.tests
```

There is **no randomness anywhere** (no seeds because no RNG). Determinism is
structural:

* Each action adds exactly one node, so a target of `N` nodes can be reached
  ONLY in exactly `N` actions. `plan(target)` is the unique post-order build.
* `count_solutions(target, alphabet)` **exhaustively enumerates all
  `|alphabet|^N` action words** of that length and asserts the count is exactly
  ONE. That is "accept exactly one solution", not a heuristic.
* `no_shorter_solution` enumerates every shorter word and asserts none reaches
  the goal.
* The fixed-state goal is first (`FixedStateGoal`); the dynamic/pattern goal
  (`ExprGoal` "value == 3") follows and its unique shortest solution is found
  exhaustively and by BFS.
* Every failure returns structured feedback `state / expected / actual /
  reason` (and an invalid action yields `last_error`, not a traceback).
* `generalization_support` reaches the same fixed goal from four different
  initial states, which is the U2 seam (bare-minimum initial state + optimal
  actions to the end state, then NEW initial states -> verify).

Checks (11): `node_type_count`, `meta_not_node_types`, `expressiveness_min4`,
`control_flow_lazy`, `fixed_goal_planner`, `unique_solution_bruteforce`,
`no_shorter_solution`, `structured_failure`, `pattern_goal`, `reductions`,
`generalization_support`.

## 8. Measured outcomes (honest)

* 11/11 checks pass on the remote venv; full output in
  `/opt/workspace/tmp/automath-new/u1/EVIDENCE.md` and
  `docs/evidence/u1/EVIDENCE.md`.
* Node types actually used by the whole suite: **4** (`Zero`, `One`, `Change`,
  `Group`). Target 4 met; the same samples also run at 3, 2 and 1 types.
* Fixed-state goals: exhaustive enumeration proves exactly ONE solution each
  (largest brute-forced target: 6 nodes).
* NOT achieved in U1 (by scope): no learned policy and no generalization
  experiment — U2 owns that; this unit only proves the environment supports it.
  The axiom library is a fixed default set (it is data, so an agent could
  extend it, but no runtime goal-synthesis is implemented). The legacy
  `env/`/`agent/` trees are untouched, so this is a parallel approach, not a
  replacement.

## 9. Prior sessions consulted

`session_search 'automath'` returned 18 sessions. The two named in the dispatch
were read: `session-f2b11485-...` and `session-c6a75787-...`; plus
`session-8cb45f58-...` (U3), `session-0e84882b-...` (U4), `session-e154a3d8-...`
(U5) and `session-e218f9e1-...`/U7. They decided: the fork lives at
`nexuslbs/automath`, main closed out at `746022d`; the symbolic RL approach did
not learn a solver on this box; the legacy full suite OOMs; work must be
small and bounded. U1 inherits all of that and adds this parallel minimal-node
branch.
