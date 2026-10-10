# LLM_LIKE_COMPARISON - automath UNIT D5 (task 4353): LLM-like learning vs the current backbone

Provenance: all numbers below are quoted from the local raw-evidence tree; none
were re-derived by re-running a campaign. Primary evidence pack:
`/opt/workspace/tmp/automath/llm-like-learning/` (campaign 1 =
`llm-like-session/`, campaign 2 = `llm-like-session2/`), design spec
`/opt/workspace/tmp/automath/llm-like-impl/docs/evolution/LLM_LIKE.md`
(1247 lines), and the baseline evidence named inline. Host written as
`<automath-host>`. No web search was used. No secret value was read or printed.
This file is not committed; a separate developer unit commits it.

Reading conventions used below:

* **C1** = `/opt/workspace/tmp/automath/llm-like-learning/llm-like-session/`
* **C2** = `/opt/workspace/tmp/automath/llm-like-learning/llm-like-session2/`
* `<s>` seeds are the repo's three-seed convention `7 / 13 / 42`.
* A path written as `C1/...` means
  `/opt/workspace/tmp/automath/llm-like-learning/llm-like-session/...`.

---

## 1. WHAT WAS BUILT

### 1.1 Mechanisms borrowed from LLM practice (and why)

Design spec `LLM_LIKE.md` section 1 (`/opt/workspace/tmp/automath/llm-like-impl/docs/evolution/LLM_LIKE.md`):

* **(a) next-step prediction** (LLM_LIKE.md:39-51) - tokens = the environment's
  derived actions (`build`/`combine`/`set`/`clear`, opt-in `pop`); a solved
  episode's witness action is the next-action cross-entropy label. Explicitly
  "NOT language modelling: there is no text, no tokenizer, no vocabulary file,
  no pretrained weights" (LLM_LIKE.md:49-51).
* **(b) attention over the last K states/actions** (LLM_LIKE.md:53-64) - exactly
  ONE single-head, single-layer attention block over `K = 8` `(state, chosen
  action)` context tokens, no FFN, no transformer stack: "It is explicitly NOT a
  transformer stack: no multi-head, no FFN, no layer-norm block stack, no
  residual depth" (LLM_LIKE.md:63-64).
* **(d) self-generated curriculum / data flywheel** (LLM_LIKE.md:66-76) - the
  training data is the agent's own solved episodes; the explicit planner/BFS is
  used only as a solvability filter, "not the training signal"
  (LLM_LIKE.md:74-76).
* **(e) small value model over `(state, next-state)`** (LLM_LIKE.md:78-85) - a
  scalar value head trained with MSE, used as the auxiliary loss.
* Explicitly NOT borrowed (LLM_LIKE.md:87-97): no transformer stack, no
  pretrained model/corpus/tokenizer, no network in the training loop, no
  differentiable engine; the environment stays the frozen stdlib `dynamic_env`.

### 1.2 Exact parameter count

* Primary config: **57,408 parameters at `d = 64`** (`LLM_LIKE.md:961-978`, and
  measured every run: `C1/train_seed_7.log` line
  `[llm_like] ACTUAL parameter count = 57408 (target <= 300000)`; also
  `param_count: 57408` in every `SUMMARY.json` and `train_summary.json`).
* Small config: **15,392 parameters at `d = 32`** (`LLM_LIKE.md:959`,
  `llm-like-impl/llm_like/tests.py:41`:
  `table = {32: 15392, 64: 57408, 128: 221312}`).
* Largest considered: 221,312 at `d = 128` (`LLM_LIKE.md:961`); target <= ~300k
  (LLM_LIKE.md:34-37).

### 1.3 Training objective

`LLM_LIKE.md` section 4 (LLM_LIKE.md:1004-1053):

```
L_policy = - (1 / N) * sum_t log softmax_j(z_j | s_t, context_t)[a_t]   (legal actions only)
L_value  = (1 / N) * sum_t ( v(s_t, s_{t+1}) - G_t )^2
L        = L_policy + lambda_v * L_value          default lambda_v = 0.5
```

Adam `lr = 1e-3`, no weight decay, 1 epoch/round, mini-batches of 256 steps
(LLM_LIKE.md:1041-1050). The config recorded in the run is
`"lambda_v": 0.5, "batch": 256, "lr": 0.001, "k_context": 8, "epsilon": 0.1`
(`C1/results/llm-like/7/train/train_summary.json`, `config`).

### 1.4 Flywheel and the solvability filter (raw counts from the training logs)

Loop: `LLM_LIKE.md:1061-1077`; solvable-only filter: `LLM_LIKE.md:1085-1102`:

* `SOLVABLE` -> keep (witness re-executed before the verdict);
* `UNSOLVABLE` -> drop before any rollout (sound proof, prohibited by
  `docs/evolution/UNSOLVABLE.md` section 6);
* `UNKNOWN` -> bounded attempt only, kept for a bounded rollout, **never counted
  as failure**: `solve_rate = solved / SOLVABLE` (LLM_LIKE.md:1097-1100).

Raw filter counts, identical for all six seed x campaign runs
(`C1/train_seed_7.log`, `.../13.log`, `.../42.log`;
`C2/train_seed_7.log`, `.../13.log`, `.../42.log`):

```
[llm_like] training pool=4 (excluded UNSOLVABLE=0) counts={'SOLVABLE': 3, 'UNKNOWN': 1, 'UNSOLVABLE': 0}
[llm_like] held-out set=160 three_way={'SOLVABLE': 140, 'UNKNOWN': 20, 'UNSOLVABLE': 0}
```

* Training pool = **4 shipped training specs** (README `C1/README`; SUMMARY
  `provenance.training_pool`: "4 shipped training specs (val64/heldout
  excluded)"). The pool is **3 SOLVABLE + 1 UNKNOWN + 0 UNSOLVABLE**, so the
  flywheel filter rule `solve_rate = solved / SOLVABLE` has a maximum numerator
  of 3 at this pool.
* Held-out pool = **0 UNSOLVABLE** per the handle rule. The eval JSON carries the
  same three-way field for every run, e.g.
  `C2/results/llm-like/42/eval/heldout_42.json`:
  `{"SOLVABLE":140,"UNKNOWN":20,"UNSOLVABLE":0}`; likewise
  `C1/results/llm-like/7/eval/heldout_7.json` etc. (all six identical).

Held-out composition (160 cases) - verified by `jq` on
`C2/results/llm-like/42/eval/heldout_42.json`:
`{fresh: 64, heldout: 2, perturb: 30, val64: 64}` = 160, matching the design
list `LLM_LIKE.md:1144-1154` (2 shipped held-out specs + `validation_pool(64)` +
fresh pool seed 424242 + 30 perturbations). Every held-out case carries a
three-way verdict in the `verdicts` field; the SOLVABLE ones are mostly
`provided_witness_replay` with `min_steps` 1-3 and the perturbations
`bounded_best_first_search` with `min_steps: 2`.

Masks: reported **separately**, mask OFF as the learned headline and mask ON as
the planner-assisted control (`LLM_LIKE.md:1160-1165`). Every `heldout_<s>.json`
has both `mask_off` and `mask_on` blocks.

---

## 2. LEARNING-OVER-TIME ANALYSIS (round 0 vs last round; no eyeballing)

Source: the snapshot curves `C1/results/llm-like/<s>/learning_curve.csv` and
`C2/results/llm-like/<s>/learning_curve.csv` (schema
`ts,round,cumulative_episodes,loss_p,loss_v,solution_rate_off,solution_rate_on,steps_off,buffer_steps,free_mib,avail_mib`).
Round 0 = first data row; "last" = last data row. The columns are exactly as the
README describes (`C1/README`).

### 2.1 Round 0 vs last round, per seed x campaign

| campaign | seed | round 0 -> last | cumulative_episodes 0 -> last | solution_rate_off 0 -> last | solution_rate_on 0 -> last | steps_off 0 -> last | loss_p 0 -> last | loss_v 0 -> last | buffer_steps last |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: |
| C1 | 7  | 0 -> 3  | 4 -> 16  | 0.5429 -> **0.4857** | 0.6071 -> **0.5571** | 1.2105 -> 1.1324 | 1.5095 -> 1.6711 | 0.2928 -> 0.1197 | 25 |
| C1 | 13 | 0 -> 3  | 4 -> 16  | 0.4786 -> **0.4714** | 0.5571 -> 0.5571 | 1.1045 -> 1.0303 | 1.4491 -> 1.1010 | 0.4527 -> 0.1764 | 23 |
| C1 | 42 | 0 -> 5  | 4 -> 24  | 0.5643 -> 0.5643 | 0.6286 -> 0.6286 | 1.5190 -> 1.3671 | 1.7381 -> 1.2729 | 0.2333 -> 0.1177 | 29 |
| C2 | 7  | 0 -> 8  | 4 -> 36  | 0.5429 -> **0.4786** | 0.6071 -> **0.5571** | 1.2105 -> 1.1493 | 1.5095 -> 1.2530 | 0.2928 -> 0.0561 | 30 |
| C2 | 13 | 0 -> 8  | 4 -> 36  | 0.4786 -> **0.4643** | 0.5571 -> **0.5500** | 1.1045 -> 1.0000 | 1.4491 -> 0.6578 | 0.4527 -> 0.0943 | 31 |
| C2 | 42 | 0 -> 22 | 4 -> 92  | 0.5683 -> **0.6187** | 0.6259 -> **0.6691** | 1.5190 -> 1.6163 | 1.7381 -> 0.5258 | 0.2333 -> 0.0755 | 94 |

(Two decimal places shown; the raw CSV holds full precision.)

### 2.2 Does the held-out solution rate improve, stay flat, or degrade?

Using the curve rows above (mask OFF is the learned headline):

* **C1 seed 7 - DEGRADES** 0.5429 -> 0.4857 (over 4 rounds / 16 cumulative episodes).
* **C1 seed 13 - DEGRADES (slightly)** 0.4786 -> 0.4714 (4 rounds / 16 episodes).
* **C1 seed 42 - FLAT** 0.5643 -> 0.5643 (6 rounds / 24 episodes).
* **C2 seed 7 - DEGRADES** 0.5429 -> 0.4786 (9 rounds / 36 episodes).
* **C2 seed 13 - DEGRADES** 0.4786 -> 0.4643 (9 rounds / 36 episodes).
* **C2 seed 42 - IMPROVES (mildly)** 0.5683 -> 0.6187 (23 rounds / 92 episodes).

Mask ON tells the same story: C1 7 down (0.6071 -> 0.5571), C1 13 flat
(0.5571 -> 0.5571), C1 42 flat (0.6286 -> 0.6286), C2 7 down
(0.6071 -> 0.5571), C2 13 down (0.5571 -> 0.5500), C2 42 up
(0.6259 -> 0.6691). So **1 of 6 runs improves, 2 are flat, 3 degrade**, and the
single improving run (C2 seed 42) moves +0.050 rate_off over 92 episodes.

The final standalone evaluation in each `SUMMARY.json` uses the full 140-case
SOLVABLE denominator and agrees with the last curve row except for C2 seed 42,
where the in-loop curve denominator is 139 (0.6187 = 86/139) and the final
standalone eval is 86/140 = **0.6143**. (C2 seed 42 is the only run whose in-loop
curve denominators are 139; all other runs use 140. This is recorded as a raw
data caveat, not smoothed over.)

Final-standalone held-out numbers (`SUMMARY.json` -> per-seed `mask_off` /
`mask_on`; `heldout_<s>.json` for the three_way field):

| campaign | seed | mask_off solved/SOLVABLE | mask_off rate | mask_on solved/SOLVABLE | mask_on rate | mask_off mean_steps | mask_on mean_steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 7  | 68/140 | 0.4857 | 78/140 | 0.5571 | 1.1324 | 1.2436 |
| C1 | 13 | 66/140 | 0.4714 | 78/140 | 0.5571 | 1.0303 | 1.1795 |
| C1 | 42 | 79/140 | 0.5643 | 88/140 | 0.6286 | 1.3671 | 1.4091 |
| C2 | 7  | 67/140 | 0.4786 | 78/140 | 0.5571 | 1.1493 | 1.2564 |
| C2 | 13 | 65/140 | 0.4643 | 77/140 | 0.5500 | 1.0000 | 1.1558 |
| C2 | 42 | 86/140 | 0.6143 | 94/140 | 0.6714 | 1.6163 | 1.5957 |

All six `heldout_<s>.json` three_way fields are
`{"SOLVABLE":140,"UNKNOWN":20,"UNSOLVABLE":0}` (e.g. `C1/.../heldout_7.json`,
`C2/.../heldout_42.json`). `solved_total` in `SUMMARY.json` is larger than
`solved_solvable` because it also counts UNKNOWN cases that happened to be solved
(e.g. C2 seed 42: `solved_solvable` 86, `solved_total` 91); the rate is over the
140 SOLVABLE.

### 2.3 Stop rule, pool size, and the loss-vs-rate divergence

* **Plateau stop.** C1 trainer config `"patience": 3`; seeds 7 and 13 logged
  `plateau stop after 4 rounds (P=3)` and seed 42 `plateau stop after 6 rounds
  (P=3)` (`C1/train_seed_*.log`). C2 used the patience-8 restart: seeds 7 and 13
  logged `plateau stop after 9 rounds (P=8)` and seed 42 `plateau stop after 23
  rounds (P=8)` (`C2/train_seed_*.log`). C2's campaign log confirms
  `wall_cap=10800s` and the run ended on the built-in plateau
  (`C2/campaign.log` final line `total_wall_s=2296`).
* **Training pool is only 4 specs, 4 episodes/round.** Every trainer snapshot
  row is `round=N episodes=4` (e.g. `C1/train_seed_7.log` lines `round=0
  episodes=4 solved=1`, ...; `C1/.../train_summary.json` `curve[]` has
  `"round_episodes": 4`), and only solved episodes are kept, so the buffer grows
  by 1-2 steps/round to a last-round `buffer_steps` of 23-94
  (`learning_curve.csv` `buffer_steps`). Cumulative kept episodes are 16/16/24
  (C1) and 36/36/92 (C2) (`SUMMARY.json` `cumulative_episodes`).
* **Loss decreases while the held-out rate does not.** `loss_v` falls in all six
  runs (e.g. C2 seed 13 0.4527 -> 0.0943; C2 seed 7 0.2928 -> 0.0561; C2 seed 42
  0.2333 -> 0.0755; C1 seed 13 0.4527 -> 0.1764), and `loss_p` falls in 5 of 6
  (C2 seed 42 1.7381 -> 0.5258; C2 seed 13 1.4491 -> 0.6578) while it rises in
  the remaining run, C1 seed 7 (1.5095 -> 1.6711).
  That is the fit the objective measures: the policy/value net fits the small
  kept buffer, not the held-out set.

---

## 3. COMPARISON TABLE

### 3.1 Explicit planner (upper-bound control, zero training)

Quoted from `/opt/workspace/tmp/automath/planner-pattern-goals/FINAL_EVIDENCE.md`
and independently re-run in
`/opt/workspace/tmp/automath/planner-pattern-goals-verification/INDEPENDENT_VERIFICATION.md`:

> `SET core33  solved=33/33 mean_actions_solved=1.6364 mean_reward=0.9182 wall=0.001s`
> `SET ext114  solved=114/114 mean_actions_solved=5.2895 mean_reward=0.7355 wall=0.006s`
> `SET unseen  solved=0/40 mean_actions_solved=None mean_reward=0.0 wall=46.360s`
> `TOTAL_WALL=46.407s`
> (`planner-pattern-goals/FINAL_EVIDENCE.md:82-88`, pattern_goals=ON.)
> The verifier reproduced `core33 33/33`, `ext114 114/114`, `unseen40 0/40`
> (`planner-pattern-goals-verification/INDEPENDENT_VERIFICATION.md:118-126`).

Before the pattern extension the fixed-state planner was
`core33 32/33, ext114 114/114` (the one miss was `expr_two/prefix1`); the pattern
path flips exactly that case to True
(`planner-pattern-goals/FINAL_EVIDENCE.md:47-92`, `:213`).

### 3.2 Evolution-selection backbone (gen-fitness, learned)

Quoted verbatim from
`/opt/workspace/tmp/automath/gen-fitness/COMPARISON.md` section 4:

| approach / flavor | core33 | ext114 | unseen40 |
| --- | ---: | ---: | ---: |
| **PLANNER** (explicit search, no training) | **32/33** | **114/114** | **0/40** |
| **GF-full** (A-target-conditioned-net, source_generation 228) | **14/33** | **12/114** | **0/40** |
| flavor B | 14/33 | 13/114 | 0/40 |
| flavor A | 12/33 | 13/114 | 0/40 |
| anchor (unchanged Phase-A agent) | 17/33 | 15/114 | 0/40 |
| G1 dense-state (flat pref, dense training) | 29/33 | 21/114 | 0/40 |

Failure mode (predecessor, quoted verbatim at `gen-fitness/COMPARISON.md:23-28`):

> "...the selection fitness is shaped_return + solved_rate_weight*solved_rate only on the
> fixed 7-state demo bundle - no term asks for held-out/new targets, so evolution
> optimizes bundle-specialized behavior."

The held-out validation term was added (`GF-full`) but "was computed but never
fired: the pool forms are unsolvable under the current action set", `val=0.000/16`
on all 228 generations (`gen-fitness/COMPARISON.md:75-83`). `unseen40 0/40` for
every arm is an eval-protocol artifact: the audit found "36/40 provably
unsolvable + 4 timeout-unknown + 0 solved"
(`gen-fitness/COMPARISON.md:137-145`).

### 3.3 Selection-methodology fix (selection-mean-rate)

From `/opt/workspace/tmp/automath/selection-mean-rate/COMPARISON.md`:

* recorded == fresh on a new seed: `1.0 (30/30)` for both `spec_dynamic_axiom`
  and `spec_multi_step` (`:39-40`);
* "shipped specs 30/30 **with the canonical mask**; still no unmasked/held-out
  generalization" (`:17`, `:47-49`);
* mask OFF collapses to `0/30` (`spec_multi_step`, `spec_dynamic_group_deep`) and
  `1-2/30` (`spec_dynamic_axiom`) (`:47-49`);
* target-change held-out `spec_multi_step_heldout` is `0/30` (`:62`).

### 3.4 Head-to-head table

Columns: solved/total, solution rate, steps, wall. Rows are the explicit planner,
the evolution-selection backbone, the selection fix, and the six llm-like runs.
"n/a" = not measured by that route/at that pool.

| # | route | pool (cases) | solved / total | solution rate | mean steps to solve | wall | params / training | source |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | explicit planner, pattern ON | core33 (33 fixed-state) | **33/33** | 1.0000 | 1.6364 (mean actions) | 0.001 s | 0 params, **zero training** | planner-pattern FINAL_EVIDENCE.md:82-88 |
| 2 | explicit planner, pattern ON | ext114 (114 fixed-state) | **114/114** | 1.0000 | 5.2895 | 0.006 s | 0 params, **zero training** | ibid. |
| 3 | explicit planner, pattern ON | unseen40 | 0/40 | 0.0000 | n/a | 46.360 s | 0 params, zero training | ibid. |
| 4 | explicit planner, base `1b8d24e` | core33 / ext114 | 32/33 / 114/114 | 0.9697 / 1.0000 | 1.6562 / 5.2895 | 0.000 s / 0.010 s | 0 params, zero training | gen-fitness COMPARISON.md:99; INDEPENDENT_VERIFICATION.md:108-114 |
| 5 | **GF-full** (evolution-selection, learned) | core33 | **14/33** | 0.4242 | n/a | wall-cap stop 7200.1 s | learned genome, 228 generations | gen-fitness COMPARISON.md:77-83, :118 |
| 6 | **GF-full** | ext114 | **12/114** | 0.1053 | n/a | (same run) | learned genome | gen-fitness COMPARISON.md:118 |
| 7 | **GF-full** | unseen40 | **0/40** | 0.0000 | n/a | (same run) | learned genome | gen-fitness COMPARISON.md:118 |
| 8 | evolution flavors A / B / anchor / G1-dense | core33 / ext114 / unseen40 | 12/33,13/114,0/40 / 14/33,13/114,0/40 / 17/33,15/114,0/40 / 29/33,21/114,0/40 | - | - | - | learned | gen-fitness COMPARISON.md:119-122 |
| 9 | selection-fix (4344), mask ON | shipped specs (`spec_dynamic_axiom`, `spec_multi_step`) | 30/30 each | 1.0000 | - | fresh re-eval PASS | 1153-gene genome | selection-mean-rate COMPARISON.md:39-40 |
| 10 | selection-fix, mask OFF | same shipped specs | 0/30, 0/30 | 0.0000 | - | - | same genome | selection-mean-rate COMPARISON.md:47-49 |
| 11 | selection-fix | `spec_multi_step_heldout` (target change) | 0/30 | 0.0000 | - | - | same genome | selection-mean-rate COMPARISON.md:62 |
| 12 | llm-like ML, C1 seed 7, mask OFF | held-out (140 SOLVABLE / 20 UNKNOWN / 0 UNSOLVABLE) | 68/140 | 0.4857 | 1.1324 (solved) | train 158.6 s; eval 17.0 s | 57,408 params, 4 rounds / 16 episodes | C1 SUMMARY.json; train_seed_7.log |
| 13 | llm-like ML, C1 seed 7, mask ON | same | 78/140 | 0.5571 | 1.2436 | eval 25.5 s | 57,408 | ibid. |
| 14 | llm-like ML, C1 seed 13, mask OFF | same | 66/140 | 0.4714 | 1.0303 | train 158.6 s; eval 17.0 s | 57,408, 4 rounds / 16 episodes | C1 SUMMARY.json |
| 15 | llm-like ML, C1 seed 13, mask ON | same | 78/140 | 0.5571 | 1.1795 | eval 25.5 s | 57,408 | ibid. |
| 16 | llm-like ML, C1 seed 42, mask OFF | same | 79/140 | 0.5643 | 1.3671 | train 223.1 s; eval 16.7 s | 57,408, 6 rounds / 24 episodes | C1 SUMMARY.json |
| 17 | llm-like ML, C1 seed 42, mask ON | same | 88/140 | 0.6286 | 1.4091 | eval 24.1 s | 57,408 | ibid. |
| 18 | llm-like ML, C2 seed 7, mask OFF | same | 67/140 | 0.4786 | 1.1493 | train 355.1 s; eval 18.2 s | 57,408, 9 rounds / 36 episodes | C2 SUMMARY.json |
| 19 | llm-like ML, C2 seed 7, mask ON | same | 78/140 | 0.5571 | 1.2564 | eval 25.2 s | 57,408 | ibid. |
| 20 | llm-like ML, C2 seed 13, mask OFF | same | 65/140 | 0.4643 | 1.0000 | train 353.5 s; eval 19.4 s | 57,408, 9 rounds / 36 episodes | C2 SUMMARY.json |
| 21 | llm-like ML, C2 seed 13, mask ON | same | 77/140 | 0.5500 | 1.1558 | eval 26.1 s | 57,408 | ibid. |
| 22 | llm-like ML, C2 seed 42, mask OFF | same | 86/140 | 0.6143 | 1.6163 | train 851.3 s; eval 15.9 s | 57,408, 23 rounds / 92 episodes | C2 SUMMARY.json |
| 23 | llm-like ML, C2 seed 42, mask ON | same | 94/140 | 0.6714 | 1.5957 | eval 24.0 s | 57,408 | ibid. |

Mask effect at the final evaluation (`mask_on.solution_rate -
mask_off.solution_rate`, all from the six `heldout_<s>.json` files): C1 seed 7
**+0.0714**, C1 seed 13 **+0.0857**, C1 seed 42 **+0.0643**, C2 seed 7
**+0.0786**, C2 seed 13 **+0.0857**, C2 seed 42 **+0.0571**. Range
**+0.057 to +0.086**, consistently positive across all six.
(The dispatch text described this as "+0.057..+0.079"; the raw `SUMMARY.json`
diffs do not support an upper bound of 0.079 - the measured maximum is +0.0857,
hit by both seed-13 runs. Reported as measured.)

### 3.5 STRICT comparability note (do not overclaim)

The pools are **different sizes, different difficulty, and different provenance**,
so the numbers in table 3.4 are NOT directly comparable across rows:

* **llm-like held-out** = 160 cases = 140 SOLVABLE + 20 UNKNOWN + 0 UNSOLVABLE.
  Its SOLVABLE forms are mostly **near-minimal**: the witness min_steps for the
  fresh/val64 cases are 1-3 and for the perturbations 2
  (`heldout_<s>.json` `verdicts` / `per_case_*`), and the learned
  `mean_steps_solved` is **1.0-1.6**. The best llm-like held-out score is
  94/140 = 0.6714 (mask ON).
* **planner core33 / ext114** are fixed-state pools of 33 and 114 cases; ext114's
  planner `mean_actions_solved` is **5.2895** - much deeper than the llm-like
  held-out forms - and the planner solves both at **rate 1.0 with zero training**
  (`planner-pattern-goals/FINAL_EVIDENCE.md:82-88`).
* **evolution unseen40** is a **mixed-unsolvability** pool: 36/40 provably
  unsolvable + 4 timeout-unknown + 0 solved, so `0/40` is an eval-protocol
  artifact, not a policy failure (`gen-fitness/COMPARISON.md:137-145`).
* **selection-fix 30/30** is on shipped specs with the canonical mask ON, a
  different harness again (`selection-mean-rate/COMPARISON.md:39-49`).

What **is** comparable: all routes run under the same "current" 4-kind action set
and the same three-way solvability rule (`unsolvable-handling/COMPARISON.md:7-23`
classifies core33/ext114/unseen40/val64 under one oracle), and the explicit
planner is a valid **upper-bound control** because it solves the fixed-state
families the learned routes are measured on. Within the llm-like evidence, the
only apples-to-apples comparisons are **mask OFF vs mask ON** at the same
checkpoint, and **round 0 vs last round** on the same 160-case held-out pool.

---

## 4. HONEST VERDICT

**The mandate question: does the LLM-like approach improve how the agent learns
over time, more efficiently than the current backbone?**

**No - not on the evidence in this tree.** The evidence, with numbers:

1. **The held-out solution rate does not rise with experience.** Across 6
   seed x campaign runs and 4-23 rounds (16-92 cumulative self-generated
   episodes), mask-OFF rate went **down in 3 runs, flat in 2, and up in 1**
   (section 2.2): C1 7 `0.5429 -> 0.4857`, C1 13 `0.4786 -> 0.4714`, C1 42
   `0.5643 -> 0.5643`, C2 7 `0.5429 -> 0.4786`, C2 13 `0.4786 -> 0.4643`,
   C2 42 `0.5683 -> 0.6187`. The one improvement is +0.050 rate_off over 92
   episodes, inside the run-to-run spread of the starting point (0.479-0.568).
   The trainers stopped on the built-in plateau rule, not on a budget
   (`C1/train_seed_7.log`: `plateau stop after 4 rounds (P=3)`;
   `C2/train_seed_42.log`: `plateau stop after 23 rounds (P=8)`).
2. **The loss fits the pool, not the task.** `loss_v` falls cleanly in all six
   runs (final values 0.0561-0.1764) and `loss_p` falls in five of six (final
   values 0.5258-1.2729), yet the held-out rate stays flat or falls. The training pool is
   **4 specs (3 SOLVABLE + 1 UNKNOWN)**, 4 episodes/round, and the flywheel only
   keeps solved episodes - so the buffer saturates on the same handful of forms
   and there is no new material for the self-curriculum to add (last-round
   `buffer_steps` 23-94). This is the concrete failure mode of the flywheel **as
   configured**: `UNSOLVABLE` is correctly dropped and `solve_rate = solved /
   SOLVABLE` is correct, but a 4-spec solvable pool cannot generate a curriculum
   that transfers.
3. **It is still far below the current backbone's best.** The best llm-like
   held-out score is **94/140 = 0.6714** (C2 seed 42, mask ON, 851 s training +
   24 s eval) on near-1-step forms (`mean_steps_solved` ~1.6). The explicit
   planner solves **core33 33/33 and ext114 114/114 (rate 1.0) with zero
   training** in **46.4 s total** (`planner-pattern-goals/FINAL_EVIDENCE.md:82-88`).
   GF-full, the learned evolution backbone, scored **14/33 and 12/114**
   (`gen-fitness/COMPARISON.md:118`) - so the llm-like net is above the learned
   backbone on its own pool but below the planner, and the pools differ
   (section 3.5), so this is a direction, not a like-for-like win. The
   planner remains the nearest-to-optimal policy for fixed-state goals and the
   learned route has not closed the gap.

**Which mechanisms are worth keeping (evidence-based):**

* **Legal-action masking - KEEP.** It is the only lever with a consistent
  positive effect: mask ON minus mask OFF is positive in all six runs,
  **+0.057 to +0.086** solution rate on the same 160-case pool
  (section 3.4), consistent with the earlier 4344 finding that the canonical
  mask is load-bearing (`selection-mean-rate/COMPARISON.md:47-49`).
* **Value-MSE objective - KEEP (trains cleanly).** `loss_v` decreases in every
  run (final values 0.0561-0.1764) with no divergence
  (`learning_curve.csv` loss_v column), and `lambda_v = 0.5` is the recorded
  config. It trains; it just does not transfer.
* **Small parameter count + cost 0 + RAM fit - KEEP.** 57,408 params at `d = 64`
  (15,392 at `d = 32`), 0.219 MiB float32 (`LLM_LIKE.md:959-978`), CPU-only,
  no network, no GPU.
* **The self-generated flywheel as configured (4-spec pool) - DO NOT KEEP as a
  transfer mechanism.** It generates no new curriculum (section 4.2) and the
  held-out curve is flat/declining. A next test would be a larger, solvable-only,
  difficulty-laddered pool (the fix `gen-fitness/COMPARISON.md:158-160` already
  named), but that is a next unit, not a result here.

Failures are reported as they are: no number in this report is masked,
hand-patched, or smoothed; the flat curve is the result.

---

## 5. COST / ENV ACCOUNTING

* **Wall time (raw).** C1 campaign `campaign_wall_s = 1252.8` s (20.9 min,
  `C1/results/llm-like/SUMMARY.json`; `FINAL_MARKER total_wall_s=1252`). C2
  campaign `campaign_wall_s = 2296.5` s (38.3 min, `C2/.../SUMMARY.json`;
  `FINAL_MARKER total_wall_s=2296`). That is ~7.0 and ~12.8 min per seed.
  Per-seed train + both eval passes (from `SUMMARY.json`):
  C1 7 = 158.6 + 17.0 + 25.5 = 201.1 s; C1 13 = 158.6 + 17.0 + 25.5 = 201.1 s;
  C1 42 = 223.1 + 16.7 + 24.1 = 263.9 s; C2 7 = 355.1 + 18.2 + 25.2 = 398.5 s;
  C2 13 = 353.5 + 19.4 + 26.1 = 399.0 s; C2 42 = 851.3 + 15.9 + 24.0 =
  891.2 s. So the per-seed work incl. eval is **3.4-14.9 min**, and the campaign
  totals (which include inter-seed overhead) are 20.9 / 38.3 min.
* **RAM.** `learning_curve.csv` `free_mib` rows span **553-873 MiB free**
  (min 553 at C2 seed 42 round 19, `C2/results/llm-like/42/learning_curve.csv`;
  max 873 at C1 seed 13 round 3, `C1/results/llm-like/13/learning_curve.csv`),
  with `avail_mib` 1387-1664. The design target box is 2 vCPU / 3814 MB / no
  swap, and the net is 0.219 MiB (`LLM_LIKE.md:34-37`, `:995-1000`); the
  buffer is bounded at `max_buffer_steps = 200000` (`LLM_LIKE.md:1119`).
* **No swap / host clean.** The same `<automath-host>` records `Swap: 0` and
  only the pre-existing discourse app / nginx-proxy / mail-relay containers, with
  no model/training container, in
  `/opt/workspace/tmp/automath/llm-like-impl/docs/evidence/u7/EVIDENCE.md`
  TASK 5 (`docker ps -a`, `free -m`, `nproc` -> 2). The learning-campaign logs
  show every trainer exited cleanly - `[campaign] seed=<s> trainer exit=0`
  for all six seeds (`C1/campaign.log`, `C2/campaign.log`) and `FINAL_MARKER`
  present in both packs - so no campaign process was left running.
  **Honest gap:** the campaign evidence packs (the `.tgz` file lists in
  `c1-ev.tgz` / `c2-ev.tgz`) contain no post-campaign `ps`/`docker ps`/`free -m`
  snapshot; the host-state proof above is from the implementation unit on the
  same host, not a post-campaign capture.
* **Cost.** **0.** Self-hosted CPU on `<automath-host>`, no cloud/GPU/paid API
  in the training loop (`LLM_LIKE.md:91-92`, "no network in the training loop"),
  matching the cost accounting of the sibling routes
  (`gen-fitness/COMPARISON.md:193-196`: host cost 0, no cloud spend).
* **Repo/secret hygiene.** No commit or push was performed by this unit; the
  trainer wrote only under `/opt/automath/llm-like-session*` and the mirror
  `/opt/workspace/tmp/automath/llm-like-learning/` (README `C1/README`). No
  credential value was read or printed; the evidence pack hashes are in
  `c1-sha256.txt` / `c2-sha256.txt`.

---

## 6. UNVERIFIED / OPEN

* No post-campaign host-state snapshot for the learning campaign (see 5).
* C2 seed 42's in-loop curve uses a 139-case denominator while the final
  standalone eval uses 140; the report quotes both (0.6187 vs 0.6143). The cause
  is not in the evidence pack.
* A larger solvable-only, difficulty-laddered flywheel pool was never run, so
  "the flywheel cannot transfer" is established only for the 4-spec pool as
  configured. Cheapest next test: build the validation pool from a
  prefix-family difficulty ladder (the fix named at
  `gen-fitness/COMPARISON.md:158-160`) and re-run the same two campaigns.
