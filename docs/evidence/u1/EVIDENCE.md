# U1 evidence: minimal-node approach (branch `new`)

Role: dsh developer. Repo: `nexuslbs/automath`. Automath host = the FIRST host
stanza in `/opt/omni/data/ssh/config`, referred to below as `<automath-host>`
(the IP is never printed). Remote clone `/opt/automath/repo`, venv
`/opt/automath/venv` (Python 3.14.4), workstation clone
`/opt/workspace/tmp/automath/repo`.

Workstation-image note: there is no `python3` in the image; code was iterated in
a throwaway `python:3.11-slim` container (read-only bind mount, `--rm`, nothing
left running) and the CANONICAL evidence was produced on the remote venv.

## 1. Mechanism choice and branch point

Chosen mechanism: **branch `new`** (the dispatch's primary option), with code in
the new package `new_approach/` so legacy `env/`/`agent/`/`test_suite/` are
untouched. Branch point = `main` HEAD:

```
branch: new
HEAD:   cf9bdf6d3954443999c7ea3a4ca88b4624b5af57   (implementation + DESIGN)
main:   746022d8d83230949b7b50fe153166e9c86fc5a6
branch-point (git merge-base new main):
        746022d8d83230949b7b50fe153166e9c86fc5a6
origin/new: cf9bdf6d3954443999c7ea3a4ca88b4624b5af57
```

The second commit on `new` (this evidence file plus the DESIGN evidence-path
fix) is reported with its sha in the U1 handoff.

## 2. `git show --stat` (implementation commit, raw on the remote clone)

```
cf9bdf6 new: minimal-node env with dynamic grouping node + deterministic single-solution tests
 docs/new-approach/DESIGN.md | 214 ++++++++++++++++++++++++
 new_approach/__init__.py    |  18 ++
 new_approach/axioms.py      | 190 +++++++++++++++++++++
 new_approach/env.py         | 312 ++++++++++++++++++++++++++++++++++
 new_approach/expressions.py | 117 +++++++++++++
 new_approach/nodes.py       | 217 ++++++++++++++++++++++++
 new_approach/reduction.py   |  31 +++
 new_approach/tests.py       | 399 ++++++++++++++++++++++++++++++++++++++++++++
 8 files changed, 1498 insertions(+)
```

## 3. ssh commands used (host redacted to `<automath-host>`)

All remote work used the typed `ssh_run(host=<automath-host>, command=...)`
tool; no raw `ssh`/`scp` binary was invoked. Key commands, verbatim:

```sh
# recon (remote pre-state: HEAD 746022d, branch main, clean)
cd /opt/automath/repo && git rev-parse HEAD && git rev-parse --abbrev-ref HEAD
  && git status --porcelain && git rev-parse origin/main
  && /opt/automath/venv/bin/python --version && nproc && free -m | head -2

# put the remote run host on the new branch
cd /opt/automath/repo && git fetch origin && git checkout -B new origin/new
  && git rev-parse HEAD && git rev-parse --abbrev-ref HEAD
  && git status --porcelain
  && git merge-base --is-ancestor 746022d8d83230949b7b50fe153166e9c86fc5a6 HEAD

# canonical test + memory bound + pytest + node-count table
cd /opt/automath/repo
/usr/bin/time -v /opt/automath/venv/bin/python -m new_approach.tests
/opt/automath/venv/bin/python -m pytest new_approach/tests.py -q
```

Remote pre-state (raw): HEAD `746022d8d83230949b7b50fe153166e9c86fc5a6`, branch
`main`, `git status --porcelain` empty, origin/main `746022d8...`, Python
3.14.4, `nproc` 2, `free -m` total 3814, available 1738, Swap 0. No `new`
branch existed (`git rev-parse --verify new` -> `fatal: Needed a single
revision`). `git merge-base --is-ancestor 746022d... HEAD` -> `YES`.

## 4. Exact test command + raw PASS output (run ON the remote clone)

Command:

```sh
cd /opt/automath/repo && /usr/bin/time -v /opt/automath/venv/bin/python -m new_approach.tests
```

Raw output:

```
[PASS] node_type_count: 4 node types used across the whole suite: Change, Group, One, Zero
[PASS] meta_not_node_types: Axiom/Goal are meta definitions, not node types (count stays 4)
[PASS] expressiveness_min4: all 11 math/logic samples evaluate correctly on the 4-node encoding
[PASS] control_flow_lazy: IF is lazy: untaken (raising) branch is not evaluated; value=7
[PASS] fixed_goal_planner: planner reaches every fixed-state goal; length == target node count
[PASS] unique_solution_bruteforce: exhaustive enumeration over the 4-action alphabet finds EXACTLY ONE solution per fixed target (max target size 6)
[PASS] no_shorter_solution: no action word shorter than the target size reaches any fixed goal
[PASS] structured_failure: wrong action -> state=[0] expected=(C(1),) actual=[0] reason=goal target is a fixed state; stack does not equal it; invalid action -> last_error='action MakeChange needs 1 nodes, stack has 0'
[PASS] pattern_goal: pattern goal value==3: unique shortest solution ('PushOne', 'MakeChange', 'MakeChange') (BFS ['PushOne', 'MakeChange', 'MakeChange'])
[PASS] reductions: same 11 samples evaluate identically with 4, 3, 2 and 1 node types (3: One:=Change(Zero); 2: Change(x):=Group(Zero,(x,)); 1: single Cell)
[PASS] generalization_support: target value=3 reached from 4 different initial states; plans=[('empty', ('PushOne', 'MakeChange', 'MakeChange')), ('one', ('MakeChange', 'MakeChange')), ('change(one)', ('MakeChange',)), ('already-goal', ())] (new-initial-state verification works for the U2 split)
RESULT: 11/11 passed in 0.030s (node_types=4, deterministic)
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.10
	Maximum resident set size (kbytes): 14044
	Exit status: 0
```

Bounded and cheap: 0.10 s wall, max RSS 14044 kB (~14 MB) on the 2 vCPU / 3.8 GB
no-swap box. No legacy test was run (it OOMs the box).

Project's own test runner also recognises the file:

```
$ /opt/automath/venv/bin/python -m pytest new_approach/tests.py -q
......                                                                   [100%]
6 passed in 0.07s
pytest_exit=0
```

## 5. Node-count table (raw, remote)

```
registered node types: ['Zero', 'One', 'Change', 'Group']
min4 types used by 11 samples: 4
min3 types used: 3
min2 types used: 2
min1 types used: 1
```

Reduction encodings that produced the row above:
`to_min3` (`One := Change(Zero)`), `to_min2` (`Change(x) := Group(Zero,(x,))`),
`to_cell` (single `Cell` container) in `new_approach/nodes.py` +
`new_approach/reduction.py`. All 11 samples evaluate to identical values in all
four encodings (the `reductions` check).

The operator's `< 3` question: a strict syntactic reading reaches 1 node type
(the pure S-expression / hereditarily-finite-set universe); `< 3` is therefore
possible unless one additionally requires two distinguishable nullary value
nodes and a typed unary-vs-n-ary distinction, under which the minimum is 4.
This is reported honestly in `docs/new-approach/DESIGN.md` section 6.

## 6. Deliverables

* Branch `new`, branch point `746022d8d83230949b7b50fe153166e9c86fc5a6`,
  implementation commit `cf9bdf6d3954443999c7ea3a4ca88b4624b5af57`, pushed to
  `origin/new`.
* `docs/new-approach/DESIGN.md` (node set, dynamic grouping node, expressiveness
  argument, reduction attempt, <=10 verification table, axiom/goal semantics,
  deterministic test plan, prior-session decisions).
* `new_approach/` (7 modules) implementing the 4-node environment.
* This file, also copied to `/opt/workspace/tmp/automath-new/u1/EVIDENCE.md`.

## 7. Honest gaps

* U1 does not train a learned policy or run the U2 generalization experiment; it
  proves the env supports the split (`plan` with non-empty initial stacks,
  `generalization_support` check). The legacy trees are untouched.
* The axiom library is a fixed default set; it is data (`AxiomSet`) so it can be
  extended, but no runtime goal synthesis is implemented.
* Secret scan of the staged diff
  (`git diff --cached | grep -nE '<credential patterns>'`) returned zero hits
  before the implementation commit; no credential value is present in any file.
* Prior sessions consulted: `session_search 'automath'` (18 hits), including the
  two named in the dispatch (`session-f2b11485-...`, `session-c6a75787-...`)
  plus U3/U4/U5/U7 sessions. They established the fork/host/venv facts, the
  `main` closeout at `746022d`, that the symbolic RL policy did not learn a
  solver, and that the legacy full suite OOMs this box.
