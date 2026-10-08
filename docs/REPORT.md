# automath improvement and alternative-approach report (units U6+U7)

Operator question: can the automath symbolic-math environment be improved so a
small self-hosted LLM (or another approach) actually solves the math questions
on a 2 vCPU / 3.8 GB no-swap box, and what is the best alternative approach?

This report is grounded only in measurements taken on the fork and on the remote
host. Every number below is either produced by a command recorded in
`/opt/workspace/tmp/automath/minicpm5-1b/EVIDENCE.md` or is a prior unit's raw
log named inline. Nothing is extrapolated silently.

Repo: `nexuslbs/automath` (fork of `lucasbasquerotto/automath`), branch `main`.
This report's base commit: `93e6ec412c5c57751eab800385afbe6c8f78645a`.

---

## 1. What was done

1. Forked `lucasbasquerotto/automath` to `nexuslbs/automath` and cloned it to
   the workstation. Origin carries upstream history (286 commits on `main`) plus
   the three fork commits below.
2. Verified the code against the source: `docs/VERIFICATION.md` (unit U2a) maps
   the state model, the action layer, the reward, the agents and the entry
   points with `file:line` and verbatim quotes.
3. Implemented the bounded/legible design (unit U3a, `docs/DESIGN.md`):
   * `1415d9c` `env: bound history, expose action errors, charge per-step reward`
     adds `agent/llm_agent.py` (the LLM seam, 181 lines), the per-episode history
     bound (`config/settings.py`, `DEFAULT_MAX_HISTORY_STATE_SIZE = 128`), the
     compact `FullState.render_observation`, `FullState.last_action_error()` and
     the per-step penalty in `env/reward.py`.
   * `93e6ec4` `scripts: per-case registry drives real math envs` adds
     `scripts/run_case.py`, the per-case runner used for every measurement here.
4. Built and ran the six-case registry under the runner's `simple` baseline, the
   official module tests under `/usr/bin/time`, and the LLM agent against a
   self-hosted `minicpm5-1b` (Q8_0) on the remote host.
5. Captured raw evidence to `/opt/workspace/tmp/automath/minicpm5-1b/`
   (`EVIDENCE.md`, `raw-selftest.txt`, `raw-llm-final.log`,
   `raw-u6u7-llm-truncated.json`).

Raw evidence pointers (on the remote host unless noted):

| evidence | pointer |
| --- | --- |
| six-case registry list + selftest | `/opt/automath/u4a/list.txt`, `/opt/automath/u4a/selftest.txt` |
| six cases under the baseline (`--agent simple`, 40 steps) | `/opt/automath/u4a/allcases.txt` and this unit's fresh re-run (EVIDENCE.md) |
| module wall/RSS | `/opt/automath/measure/summary.txt` |
| LLM agent run + bounded probe | `/root/llm-final.log`, `/root/llm_probe.py` |
| model deploy log | `/root/llm-deploy-deploy.log` |
| workstation copy | `/opt/workspace/tmp/automath/minicpm5-1b/EVIDENCE.md` |

---

## 2. VERDICT

**No good result. The agent does not solve the math questions.**

On the base commit `93e6ec4`, with the runner's own `simple` policy and
`--max-steps 40`, all six real cases end `goal=False` (fresh run this unit;
`max_history=40`, `dropped_history=0`):

| case | allowed actions | wall s | reward | goal |
| --- | ---: | ---: | ---: | --- |
| binary_int | 22 | 0.611055 | -4040.0 | False |
| signed_int | 22 | 0.547026 | -4040.0 | False |
| int_to_binary | 22 | 1.372998 | -3958.545217 | False |
| boolean_lt | 22 | 1.764909 | -3876.894827 | False |
| indices | 34 | 2.559794 | -3308.389211 | False |
| control_flow | 34 | 7.956449 | -3323.513677 | False |

The reward floor of `-4040.0` is exactly `40 * -101`: every one of the 40 steps
was an invalid action (the `-100` last-step-error reward plus the `-1`
step penalty), so most steps produce a typed error row and never change the
math state.

The LLM agent does not do better. With the shipped 30 s client timeout
(`agent/llm_agent.py:80`) and the agent's own 757-token prompt, step 1 alone
takes longer than the timeout and the run aborts with
`LlmUnavailableError ... timed out` at `wall_ms=30059.623` (raw:
`/root/llm-final.log:3`). The bounded probe (60 s client timeout) gets replies
but still scores `goal=False` (`/root/llm-final.log:38-110`).

The measured environment is honest and correct: the goal already lives in the
state (`env/full_state.py:463-472`), actions return a new `FullState`, and an
invalid action is a typed error row rather than a crash
(`env/action.py:442-449`). The failure is in the *agents* and in the *cost of
each step*, not in the math semantics.

---

## 3. Root causes

Each cause below carries its evidence and the `file:line` that produces it.

### (i) The default agent is an untrained/fixed policy, not the trained RL policy

The runner's `simple` baseline is `_SimplePolicy`
(`scripts/run_case.py:208-234`), a **seeded uniform random explorer** over the
allowed action catalogue and arguments (`arg1/arg2 in 0..2`, `arg3 in 0..1`).
It is not a learner at all: its docstring says "deliberately not a solver: it is
the arbitrary agent baseline". That is the policy behind the six `goal=False`
rows above.

The repo's own default learner is also untrained and not the torch RL agent:

* `config/agent_settings.py:23` sets `AGENT_TYPE: str = "simple"`, so root
  `train.py` constructs the NumPy `SimpleAgent` with random weight init
  (`agent/simple_agent.py:69-88`) and no checkpoint is loaded.
* The torch DQN `SmartAgent` (`agent/smart_agent.py:261-326`) is deliberately
  disabled: the import is commented out (`train.py:16`) and the branch raises
  `NotImplementedError` (`train.py:73-74`). `torch` is not in
  `requirements.txt` (see `docs/VERIFICATION.md` section 11).

So nothing in the shipped default path ever learns a solution.

### (ii) The action space is far too large for a 1B model, and every wrong action costs -101

`run_case.py --list` reports 22 allowed actions for the four arithmetic/boolean
cases and 34 for `indices`/`control_flow` (24 basic actions are emitted in all
cases). The curated set is `ESSENTIAL_ACTIONS` (`env/node_types.py:31-54`), and
each action carries three integer arguments whose meaning is type-dependent
(`env/action.py:602-612`). A small model must pick the right typed action *and*
the right three integers from a rendered symbolic state.

The penalty for a miss is steep: `env/reward.py:61-62` returns
`-100 - step_penalty`, and the default `step_penalty` is `1.0`
(`env/reward.py:36`), so each invalid action costs `-101`. The observed
`-4040.0` is 40 such steps. There is no cheap exploration signal for a
scale-1B model here.

### (iii) Per-step LLM latency exceeds the client timeout, so zero usable LLM steps

The shipped agent hardcodes `timeout: float = 30.0`
(`agent/llm_agent.py:80`) and appends `/v1/chat/completions` itself
(`agent/llm_agent.py:98`). Measured on the remote host:

* The agent's own prompt (2550 chars, 757 prompt tokens) produced ~2070
  completion tokens in **289.1 s** before the `content` field appeared (prior
  unit measurement, briefing).
* The literal CLI run still timed out at step 1: `wall_ms=30059.623`
  (`/root/llm-final.log:3`).
* The bounded probe with a 60 s client timeout recorded per-step LLM latencies
  of 23.8 / 42.8 / 43.2 s (binary_int), 23.6 / 23.5 / 23.4 s (boolean_lt) and
  57.8 / 48.4 / 48.4 s (control_flow) (`/root/llm-final.log:58,82,106`).
* A minimal fresh request this unit (85 prompt tokens, 277 completion tokens)
  still took 23.2 s of generation at **11.89 tok/s** (EVIDENCE.md).

So a single step costs tens of seconds to minutes, while the budget is 30 s:
the shipped configuration yields zero usable LLM steps.

**Correction to an earlier claim.** An earlier unit claimed the model's
`content` is always empty. That was refuted by the independent tester and is
refuted again by the fresh raw capture: the model is a reasoning model that
writes `reasoning_content` first (967 chars in the fresh capture) and then emits
valid `content`:

```json
{"action": "CreateScratch", "args": [1, 2, 3]}
```

`CreateScratch` is an allowed action, the reply is valid JSON, and there is **no
parse-level defect**. The blocker is per-step latency plus the 30 s timeout.

### (iv) Harness-level retention blows the 3.7 GB no-swap box before the RL loop can train

The full suite cannot complete on this box. `arithmetic_test` is killed at the
300 s timeout with maximum RSS **1805.7 MiB** (`/opt/automath/measure/summary.txt`:
`arithmetic_test exit=124 wall=301s` / `Maximum resident set size (kbytes):
1849020`). Uncapped it is OOM-killed at ~1.93 GB (briefing). All other modules
pass: `basic_test` 8.02 s / 131244 kB; `boolean_test` 22.75 s / 198848 kB;
`indices_test` 15.04 s / 171784 kB; `control_flow_test` 23.26 s / 149516 kB;
`action_00..05` 4.31 to 33.38 s / 99848 to 229040 kB
(`/opt/automath/measure/summary.txt`).

Crucially, the per-episode history bound did **not** move the number: after
setting `DEFAULT_MAX_HISTORY_STATE_SIZE = 128` the RSS was unchanged at
**1797.8 MiB** (independent tester measurement, briefing). That isolates the
retention to the **test harness**, not the environment:

* `test_utils.run_test` /
  `run_module_test` return the full result object and keep it alive for the
  duration of the call (`test_suite/test_utils.py:5-27`).
* `arithmetic_test.test()` accumulates **every** sub-case's returned
  `final_states` into one list (e.g. `final_states += ...` across
  `test_suite/arithmetic_test.py:7549` and the tail at `:7560-7582`), so all 27
  sub-case `FullState` trees stay reachable until the module returns.

Because the box has no swap (`free -m` shows `Swap: 0`), the OOM kill arrives
before a long RL training loop could even start.

### (v) The repo has no LLM seam at all

The original repo contains zero LLM/network code; the only third-party imports
are `numpy`, `sympy` and `torch` (`docs/VERIFICATION.md` sections 10 and 11).
The "small LLM as a smarter agent" path had to be written from scratch
(`agent/llm_agent.py`, added in `1415d9c`). That is why the seam is thin and
un-tuned: it renders the whole state, lists all allowed actions and hopes a 1B
model picks one, without grammar constraints or a compact observation.

### Secondary defect: the `--base-url` convention

`LlmAgent.__init__` only `rstrip('/')`s the base URL
(`agent/llm_agent.py:83-89`) and `_chat` always appends `/v1/chat/completions`
(`agent/llm_agent.py:98`). `--base-url` must therefore be the host root
(`http://localhost:8080`); passing `.../v1` would build
`/v1/v1/chat/completions` and fail. The deploy tool prints the OpenAI base as
`http://localhost:8080/v1` (`/root/llm-deploy-deploy.log`), which is the
opposite convention, so this should be normalised or documented.

---

## 4. Concrete next fixes and their expected impact

Ordered by value per unit of work on this box.

1. **Grammar-constrained decoding (llama.cpp GBNF).** Restrict the reply to the
   allowed action schema (`{"action": <enum>, "args": [int,int,int]}`). This
   removes the reasoning-token blowup that dominates latency and eliminates the
   parse risk entirely. Expected: a valid, schema-conformant action on every
   step, with completion tokens measured in tens rather than ~2070. This is the
   single highest-leverage fix for the LLM path.
2. **A hierarchical/macro action layer.** Replace the 22 to 34 typed actions
   with a handful of macro-actions plus a tiny DSL. Expected: the selection
   problem shrinks from 34 types times 3 integers to a few dozen macro choices,
   which is plausible for a 1B model and much cheaper to search.
3. **A compact, bounded observation (a few hundred tokens).** Today the
   observation is the rendered state (19 nodes for arithmetic, 1997 for
   control_flow; `--selftest`). Expected: prompt tokens drop by an order of
   magnitude, cutting both prefill and the attention cost of the reasoning
   trace.
4. **Enable and train the repo's own torch policy** as the action selector,
   using the 128-entry bounded history and the fewest-actions reward, with the
   LLM used only offline as a prior/heuristic. Expected: no LLM in the training
   loop, so wall time is CPU-bound and can actually run on 2 vCPU. Requires
   adding `torch` (already declared in `requirements-train.txt`).
5. **Fix the harness accumulation.** Drop or aggregate `final_states` per case
   instead of holding all 27 (`test_suite/arithmetic_test.py:7549-7582`,
   `test_suite/test_utils.py:5-27`). Expected: `arithmetic_test` drops from
   > 1805.7 MiB / OOM at 1.93 GB to a bounded footprint, letting the module
   complete.
6. **Raise the client timeout and cap `max_tokens` (or disable thinking).** A
   single step should cost seconds, not minutes. Expected: with a
   non-reasoning or thinking-disabled 1B and `max_tokens` capped at a few
   hundred, per-step latency falls to a few seconds, inside a sane timeout.
   Without fixes 1 and 3, this alone is not enough.

---

## 5. Alternative approaches

The operator asked whether a different approach is better. Four options,
scored for a **2 vCPU / 3814 MB / no-swap** box.

### (A) LLM as the online action selector

Requires grammar-constrained decoding, short prompts and either a much
bigger/faster host or a non-reasoning 1B. Feasibility on this box: poor today.
The measured 1B spends ~277 (minimal prompt) to ~2070 (agent prompt) completion tokens per decision at
**11.89 tok/s**, and the box has ~1.1 GB available with the model loaded
(`free -m` this unit: 222 MB free, 1151 MB available, swap 0; model
631.5 MiB idle and 1.498 GiB under load inside a 3 GiB cap). A 40-step episode
would take many minutes even after grammar constraints, and training is out of
the question. Expected impact: viable only as a *heuristic*, not as the
per-step controller.

### (B) The repo's own symbolic RL policy trained on the bounded env

Best fit for THIS box. The environment is pure Python and CPU-only; the torch
DQN already exists (`agent/smart_agent.py`), the history is bounded to 128 and
the reward now charges per step. No LLM is in the loop, so the 2 vCPU machine
can actually run episodes to completion once the harness retention (fix 5) is
fixed. Cost: add `torch` (already declared in `requirements-train.txt`), enable
the `smart` branch in `train.py`, and run a bounded training loop. Expected
impact: the only path that can plausibly learn a solver on this hardware, though
training time on 2 vCPU is still the main risk.

### (C) Hybrid: LLM proposes macro-action candidates offline, RL selects online

The LLM is used once (offline) to mine a small set of promising macro-actions
per goal family; the repo's RL policy (or a small search) selects among them
online. Feasibility: good. The expensive model latency is paid once per macro,
not once per environment step, and the online controller stays CPU-only. On the
same box the offline LLM pass is affordable (the fresh minimal call took 23.2 s
for 277 tokens); grammar constraints make it safer. Expected impact: a middle
ground that keeps the LLM's value without paying per step.

### (D) Guided search (DFS/MCTS) over the symbolic state, LLM as heuristic evaluator

The symbolic environment is deterministic and typed, so a systematic search can
run without any learned policy. The LLM is used only to score candidate states.
Feasibility on 2 vCPU: moderate. Search is CPU-bound and the branching factor is
the same 22 to 34 actions; with the compact observation and macro layer
(fixes 2 and 3) it is tractable for the atomic arithmetic cases but the
control_flow state (1997 nodes) is expensive. Expected impact: works best as a
sanity oracle for the small cases, not as a general solver.

### Recommendation: ONE next step

**Do (B), the repo's own symbolic RL policy, but first land fix 5 (drop the
harness-level `final_states` accumulation) and then fix 2 and 4 together**
(macro-action layer plus enabling the torch `SmartAgent`). This is the only
combination that fits 2 vCPU / 3.8 GB with no GPU and no network in the training
loop, and it is the shortest path to a policy that actually solves the six
cases. Keep the LLM offline as a macro-action prior (option C) once the online
controller is cheap; do not attempt option (A) as the per-step controller on
this host.

---

## 6. Reproduction and cost accounting

All commands run on the remote host from `/opt/automath/repo` with the venv at
`/opt/automath/venv`. Every command was time-boxed.

```sh
# deploy the small model (llm-deploy)
cd /opt/llm-deploy && ./llm-deploy ...        # -> /root/llm-deploy-deploy.log
# health: http://localhost:8080/health

# registry list and selftest
python scripts/run_case.py --list
python scripts/run_case.py --selftest

# six cases under the baseline, 40 steps each
python scripts/run_case.py --case binary_int    --agent simple --max-steps 40
python scripts/run_case.py --case signed_int    --agent simple --max-steps 40
python scripts/run_case.py --case int_to_binary --agent simple --max-steps 40
python scripts/run_case.py --case boolean_lt    --agent simple --max-steps 40
python scripts/run_case.py --case indices       --agent simple --max-steps 40
python scripts/run_case.py --case control_flow  --agent simple --max-steps 40

# LLM agent (the shipped 30 s timeout fails at step 1)
python scripts/run_case.py --case binary_int --agent llm \
    --base-url http://localhost:8080 --model minicpm5-1b --max-steps 8

# bounded probe with a raised timeout, no repo change
python /root/llm_probe.py binary_int 3 http://localhost:8080 minicpm5-1b 60 256

# one raw chat completion
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minicpm5-1b","messages":[{"role":"system","content":"You choose the next action for a symbolic-math environment. Reply with a single JSON object and nothing else: {\"action\": \"<ActionTypeName>\", \"args\": [int, int, int]}."},{"role":"user","content":"Goal: HaveResultScratch(Eq[BinaryInt]). Allowed actions include CreateScratch. Reply with the next action as JSON."}],"max_tokens":2048,"temperature":0}'

# module wall/RSS (the prior unit's runner, timeout 300 s)
/usr/bin/time -v timeout 300 python -u -c "from test_suite import arithmetic_test as m, test_utils; test_utils.run_module_test(m.test)"
# see /opt/automath/measure/summary.txt for the recorded wall/RSS
```

Measured cost of the LLM path, **reported as-is**:

* Fresh raw call this unit: `prompt_tokens=85`, `completion_tokens=277`,
  `total_tokens=362`; `predicted_n=277` in `predicted_ms=23216.805`, i.e.
  **11.888 tok/s**; `finish_reason=stop`; `reasoning_content` 967 chars,
  `content={"action": "CreateScratch", "args": [1, 2, 3]}`.
* Agent's own prompt (prior unit): ~757 prompt tokens, ~2070 completion tokens
  in 289.1 s.
* **Cost: 0.** The model is self-hosted on the remote host, so there is no
  per-token billing. The meaningful cost is wall time (23.2 s for one tiny
  decision at 11.89 tok/s) and the memory footprint (631.5 MiB idle,
  1.498 GiB under load, inside a 3 GiB cgroup cap).

Resource state at the time of measurement: `free -m` total 3814 MB,
`Swap: 0`; before the case run 118 MB free / 1161 MB available, after
222 MB free / 1151 MB available.

---

## 7. Limits of this report

* The `arithmetic_test` OOM and the module wall/RSS numbers come from the prior
  unit's raw logs (`/opt/automath/measure/summary.txt`); this unit re-ran the
  six registry cases and the LLM call, not the full module suite.
* The 289.1 s agent-prompt latency and the ~1.93 GB uncapped OOM figure are
  prior-unit measurements quoted from the briefing and the probe log, not
  re-measured here.
* The `--base-url` `/v1` mismatch is read from the code and the deploy log; a
  `.../v1` 404 was not separately re-measured this unit.
* The full suite was not completed on this box. It cannot be, before the
  harness retention in cause (iv) is fixed.
