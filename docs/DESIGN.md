# automath design (unit U3a)

Short design note for the U3a changes: bounded state, legible failures,
fewest-actions reward, timing hooks and the small-LLM seam. It complements
`docs/VERIFICATION.md` (the file:line map of the original gaps).

## Three constraints

1. **Memory.** `FullState.history` grew without bound: `max_history_state_size`
   was accepted by `GoalEnv` but never defaulted, so
   `env/action.py:488-489` never fired. The arithmetic module reached ~1.80 GB
   RSS and the full suite was OOM-killed at 1.93 GB on a 3.7 GB no-swap box.
2. **Legibility.** An invalid action already produced a typed error row
   (`env/action.py:442-449`) but the agent could not read the cause; the state
   had no compact observation.
3. **Credit assignment.** `DefaultRewardEvaluator` returned
   `+10000` on the goal, `-100` on a last-step error and `-log(final_cost+1)`
   otherwise, so two paths to the same terminal state scored the same: action
   count was invisible.

## Encoding (what is now representable, and where)

* **Goal**: already in the state (`MetaInfo.goal`,
  `State.meta_info.goal_achieved`; `env/full_state.py`). Unchanged.
* **Bounded history**: a summary entry is a normal `HistoryNode` whose
  `action_data` carries a real `HistorySummaryActionData` node
  (`env/full_state.py`) with `dropped_count` and `dropped_cost`, so a truncated
  history still reports what it forgot. (The type system forbids subclassing an
  instantiable node, so the marker is a sibling `BaseActionData` subclass.)
* **Failure**: `BaseActionData.exception` already held the typed error info.
  `FullState.last_action_error()` now exposes it as
  `(action_type_name, error_class_name, message, action_index)`, and
  `FullState.render_observation(max_nodes=...)` renders a bounded text view
  (goal, current state, cost, last error, dropped-history count).
* **Action count**: not stored in the state; it is the number of retained
  history entries plus dropped_count, and it is charged per step by the reward
  evaluator.

## History bound and retention rule

* `config/settings.py: DEFAULT_MAX_HISTORY_STATE_SIZE = 128`.
* `GoalEnv` uses that value when the caller passes `max_history_state_size=None`
  (`env/goal_env.py`), so the existing truncation branch always has a bound.
* When `len(history) > bound`, `_apply_history_bound` (`env/action.py`)
  retains the goal/initial entry, one `HistorySummaryNode` and the most recent
  `bound - 2` entries, and folds any previous summary into the new one (so
  `dropped_count` is cumulative).
* **Why 128**: the unbounded arithmetic run grew to ~1.80 GB; retaining O(128)
  states per episode keeps per-episode history in the low-MB range while still
  leaving an agent ample recent context. The bound fires through the existing
  `env/action.py` truncation branch, not around it.

## Reward formula

`env/reward.py`, `DefaultRewardEvaluator.create(goal_reward=10000, step_penalty=1.0)`:

```
if goal achieved:        R = goal_reward - step_penalty          # +10000 - p
elif last step error:    R = -100 - step_penalty
else:                    R = -log(final_cost + 1) - step_penalty
```

Step penalty is stored as milli-units (`Integer` cannot hold a float). With the
default `step_penalty = 1.0`, three actions to a terminal state score strictly
higher than five actions to the same state, because the 5-action path pays two
extra `-step_penalty` terms and two extra non-positive `-log(cost+1)` terms.
`+goal_reward` remains dominant. `step_penalty` defaults to `1.0` when the
argument is absent (a legacy one-argument evaluator still evaluates).

## Timing hooks

`Environment.step` (`env/environment.py`) records `time.perf_counter()` around
`action.run_action` and appends to `Environment.step_times`; `total_time()`
sums them. `reset()` clears them. Nothing else changes when unused.

## LLM seam

`agent/llm_agent.py` is a `BaseAgent`-compatible `LlmAgent`:

* renders the observation with `FullState.render_observation`,
* lists allowed action type names and their `action_index`,
* POSTs an OpenAI-compatible `{base_url}/v1/chat/completions` request with
  `model` using **stdlib `urllib.request`** (no requests/openai),
* parses the reply defensively as `{"action": "<TypeName>", "args": [...]}`;
  unknown or unparseable replies return `RawAction.with_raw_args(0, 0, 0, 0)`,
  which the environment records as a typed error row rather than crashing.
* When no endpoint is reachable it raises `LlmUnavailableError` with the URL
  and the underlying error.
* `python -m agent.llm_agent` prints the EXACT request (URL + JSON payload) it
  would send, so the prompt can be verified without an LLM running.

## Commands

```sh
# design tests
python -m pytest test_suite/design_test.py -v
python test_suite/design_test.py

# existing tests that must keep passing
python -c "from test_suite import basic_test, boolean_test, indices_test; basic_test.test(); boolean_test.test(); indices_test.test(); print('OK')"

# runner
python scripts/run_case.py --agent simple --max-steps 10
python scripts/run_case.py --agent llm --base-url http://localhost:8000 --model local-model --max-steps 5 --json

# prompt selftest (no LLM needed)
python -m agent.llm_agent
```

## Dependency truth

`torch` is used only by `agent/smart_agent.py:6-8` and is deliberately not in
`requirements.txt`. It is declared in `requirements-train.txt` (with the
reason); install it only to run the DQN SmartAgent.
