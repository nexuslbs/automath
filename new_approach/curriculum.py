"""Flavor B: difficulty curriculum over the G1 dense-state training distribution.

Operator intent (thread 4331, quoted verbatim):

    "start with a very simple environment, make it more complex as the agent
     learns"

Flavor B keeps flavor A's policy genome (the flat ``PolicyNet`` weights in
``Genome.net``), the shaped reward (``evolution_rewards.RewardConfig``) and the
fitness formula (``shaped_return + solved_rate_weight * solved_rate``) exactly
as they are.  The ONLY difference is the TRAINING-STATE DISTRIBUTION: the G1
dense-state generator (``gen_common.dense_cases``: 320 cases = 77 core + 243
evo) is partitioned into three difficulty tiers and each generation samples its
per-generation training bundle from the CURRENT phase's tier(s).

Schedule rule (documented choice): a FIXED generation schedule.  The phase is a
pure function of the generation index (``CurriculumSchedule.phase_index``), so
the run is exactly reproducible and a resumed run continues the same curriculum
without storing any gate state.  The adaptive solved-rate alternative was
rejected because flavor A's best solved rate on the demo bundle is <= 1/5, so a
solved-rate gate would stall the curriculum in tier 1 for the whole bounded
wall-clock window; the fixed schedule guarantees all three tiers are trained
inside the 2 hour detached run.

Tier definitions (deterministic, from dense-case properties):

* tier 1 EASY   - direct non-trivial prefixes on SIMPLE targets
                  (``/prefix`` with stack depth <= easy_prefix_max_depth,
                  stack nodes <= easy_prefix_max_nodes and
                  target size <= easy_prefix_max_target_size).
* tier 2 MEDIUM - structured perturbations (``/pert``) that are not yet deep,
                  plus prefixes that outgrew tier 1 but are still shallow.
* tier 3 HARD   - random intermediate stacks (``...random...``) and every
                  deep / many-node / complex-target case.

Reachability filter: of the 320 G1 dense cases, 34 CORE cases target a
``Group`` whose ``(tag, arity)`` has no build action in the extended action
alphabet (for example ``G(1;0)`` and ``G(1;1,1,1)``); those targets are
structurally impossible for the ``EvoEnv`` the bundle trains on, so they are
excluded.  The remaining 286 cases are reachable from the sampled stacks.

Pure standard library.  All thresholds and phase lengths come from the
``curriculum`` block of the run config (``config/evolution_targetB.json``).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from . import gen_common as gc
from .evolution_bundle import Bundle, BundleState, target_feasible
from .evolution_run import SimpleCase
from .nodes import size

TIER_EASY = 1
TIER_MEDIUM = 2
TIER_HARD = 3
TIER_ORDER: Tuple[int, ...] = (TIER_EASY, TIER_MEDIUM, TIER_HARD)

DEFAULT_THRESHOLDS: Dict[str, int] = {
    "easy_prefix_max_depth": 2,
    "easy_prefix_max_nodes": 4,
    "easy_prefix_max_target_size": 6,
    "medium_max_depth": 3,
    "medium_max_nodes": 12,
    "medium_max_target_size": 22,
}

#: Default phase plan used when a config omits ``phases``.  The last phase is
#: unbounded in practice (a very large generation count) so ``phase_index``
#: stays a pure function.
DEFAULT_PHASES: List[dict] = [
    {"name": "easy", "tiers": [TIER_EASY], "generations": 30, "batch": 12},
    {"name": "medium", "tiers": [TIER_EASY, TIER_MEDIUM], "generations": 60,
     "batch": 16},
    {"name": "hard", "tiers": [TIER_EASY, TIER_MEDIUM, TIER_HARD],
     "generations": 100000, "batch": 20},
]


def _norm_thresholds(thresholds: Optional[Dict[str, int]]
                     ) -> Dict[str, int]:
    out = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        for key, val in thresholds.items():
            if key in out:
                out[key] = int(val)
    return out


def case_kind(name: str) -> str:
    """The dense-case family encoded in its generated name."""
    if "random" in name:
        return "random"
    if "/pert" in name:
        return "pert"
    if "/prefix" in name:
        return "prefix"
    return "other"


def difficulty(case: SimpleCase, kind: Optional[str] = None) -> float:
    """A deterministic scalar difficulty used to CHECK tier ordering.

    It is monotone in stack depth, stack node count, target size and the case
    family (prefix < perturbation < random).
    """
    kind = kind or case_kind(case.name)
    stack = tuple(getattr(case.env, "initial_stack", ()))
    depth = len(stack)
    nodes = sum(size(n) for n in stack)
    target = getattr(case.env.goal, "target", None)
    tsize = size(target) if target is not None else 0
    family_bonus = {"prefix": 0.0, "pert": 1.0, "random": 2.0}.get(kind, 2.0)
    return depth + nodes / 5.0 + tsize / 10.0 + family_bonus


def assign_tier(case: SimpleCase, thresholds: Optional[Dict[str, int]] = None
                ) -> int:
    """Deterministic difficulty tier (1 easy / 2 medium / 3 hard)."""
    th = _norm_thresholds(thresholds)
    kind = case_kind(case.name)
    stack = tuple(getattr(case.env, "initial_stack", ()))
    depth = len(stack)
    nodes = sum(size(n) for n in stack)
    target = getattr(case.env.goal, "target", None)
    tsize = size(target) if target is not None else 0

    if kind == "random":
        return TIER_HARD
    if kind == "pert":
        if (depth <= th["medium_max_depth"]
                and nodes <= th["medium_max_nodes"]):
            return TIER_MEDIUM
        return TIER_HARD
    # prefix (or an unexpected family): easy when short/shallow/simple.
    if (depth <= th["easy_prefix_max_depth"]
            and nodes <= th["easy_prefix_max_nodes"]
            and tsize <= th["easy_prefix_max_target_size"]):
        return TIER_EASY
    if (depth <= th["medium_max_depth"]
            and nodes <= th["medium_max_nodes"]
            and tsize <= th["medium_max_target_size"]):
        return TIER_MEDIUM
    return TIER_HARD


_TIER_CACHE: Dict[Tuple, Tuple[Dict[int, List[SimpleCase]], int]] = {}


def _tiered_raw(thresholds: Optional[Dict[str, int]]
                ) -> Tuple[Dict[int, List[SimpleCase]], int]:
    """(tiers, excluded_infeasible) for a threshold set, computed once."""
    th = _norm_thresholds(thresholds)
    key = tuple(sorted(th.items()))
    cached = _TIER_CACHE.get(key)
    if cached is None:
        core, evo, _total = gc.dense_cases()
        tiers: Dict[int, List[SimpleCase]] = {
            TIER_EASY: [], TIER_MEDIUM: [], TIER_HARD: []}
        excluded = 0
        for case in list(core) + list(evo):
            target = getattr(case.env.goal, "target", None)
            if target is not None and not target_feasible(target)[0]:
                # unreachable in the EvoEnv the bundle trains on
                excluded += 1
                continue
            tiers[assign_tier(case, th)].append(case)
        cached = (tiers, excluded)
        _TIER_CACHE[key] = cached
    return cached


def tiered_cases(thresholds: Optional[Dict[str, int]] = None
                 ) -> Dict[int, List[SimpleCase]]:
    """Partition the reachable G1 dense cases into ``{tier: [case, ...]}``.

    Cached per threshold set; a copy of each tier list is returned so callers
    cannot corrupt the cache.  Cases whose target has no build action in the
    EVO alphabet are excluded (see ``excluded_infeasible``).
    """
    tiers, _excluded = _tiered_raw(thresholds)
    return {tier: list(cases) for tier, cases in tiers.items()}


def excluded_infeasible(thresholds: Optional[Dict[str, int]] = None) -> int:
    """How many dense cases were dropped for being unreachable in EvoEnv."""
    return _tiered_raw(thresholds)[1]


def sample_cases(cases_by_tier: Dict[int, Sequence[SimpleCase]],
                 tiers: Sequence[int], batch: int,
                 rng: random.Random) -> List[SimpleCase]:
    """Draw ``batch`` cases (without replacement) from the union of ``tiers``.

    ``batch <= 0`` or a pool smaller than ``batch`` returns the whole pool, so
    the result is always non-empty when any listed tier is non-empty.
    """
    pool: List[SimpleCase] = []
    for tier in tiers:
        pool.extend(cases_by_tier[int(tier)])
    if batch <= 0 or batch >= len(pool):
        return list(pool)
    return rng.sample(pool, batch)


def cases_to_bundle(cases: Sequence[SimpleCase], name: str, total_budget: int,
                    thresholds: Optional[Dict[str, int]] = None) -> Bundle:
    """Turn sampled dense cases into a multi-state ``Bundle`` the shaped
    controller/loop can train on (same ``BundleState`` surface as the demo
    bundle, so ``evolve_one_generation_shaped`` and the fitness formula are
    untouched)."""
    th = _norm_thresholds(thresholds)
    states: List[BundleState] = []
    for i, case in enumerate(cases):
        target = getattr(case.env.goal, "target", None)
        if target is None:
            continue
        ok, why = target_feasible(target)
        tier = assign_tier(case, th)
        states.append(BundleState(
            sid="d%02d" % i,
            name=case.name,
            target=target,
            feasible=bool(ok),
            initial_stack=tuple(getattr(case.env, "initial_stack", ())),
            expected_len=0,
            level="tier%d" % tier,
            impossible_reason=("" if ok else why),
        ))
    return Bundle(name, tuple(states), total_budget)


class CurriculumSchedule:
    """A fixed generation-schedule curriculum.

    ``phase_index(gen)`` is a pure function of the 0-based generation counter:
    phase ``i`` covers ``[sum(lengths[:i]), sum(lengths[:i+1]))``.  The default
    rule is ``generation_schedule``; the class rejects any other rule so the
    shipped behavior cannot silently drift.
    """

    def __init__(self, phases: Optional[Sequence[dict]] = None,
                 thresholds: Optional[Dict[str, int]] = None,
                 seed: int = 20261009,
                 rule: str = "generation_schedule") -> None:
        if rule != "generation_schedule":
            raise ValueError("unsupported curriculum rule: %r" % (rule,))
        self.rule = rule
        self.phases = [dict(p) for p in (phases or DEFAULT_PHASES)]
        self.thresholds = _norm_thresholds(thresholds)
        self.seed = int(seed)
        self._validate()
        self.cases_by_tier = tiered_cases(self.thresholds)

    def _validate(self) -> None:
        if not self.phases:
            raise ValueError("curriculum phases must be non-empty")
        for i, ph in enumerate(self.phases):
            if not ph.get("tiers"):
                raise ValueError("phase %d has no tiers" % i)
            for tier in ph["tiers"]:
                if int(tier) not in TIER_ORDER:
                    raise ValueError("phase %d has bad tier %r" % (i, tier))
            if int(ph.get("generations", 0)) <= 0:
                raise ValueError("phase %d needs generations > 0" % i)
            if int(ph.get("batch", 0)) <= 0:
                raise ValueError("phase %d needs batch > 0" % i)

    @classmethod
    def from_config(cls, data: Optional[dict], seed: int
                    ) -> Optional["CurriculumSchedule"]:
        """Build a schedule from an ``EvoConfig.curriculum`` dict.

        ``None`` or ``{"enabled": false}`` disables the curriculum entirely
        (flavor A behavior).
        """
        if not data or not data.get("enabled", True):
            return None
        return cls(phases=data.get("phases") or DEFAULT_PHASES,
                   thresholds=data.get("thresholds"),
                   seed=int(data.get("seed", seed)),
                   rule=data.get("rule", "generation_schedule"))

    def phase_index(self, gen: int) -> int:
        g = max(0, int(gen))
        acc = 0
        for i, ph in enumerate(self.phases):
            acc += int(ph["generations"])
            if g < acc:
                return i
        return len(self.phases) - 1

    def phase(self, gen: int) -> dict:
        return self.phases[self.phase_index(gen)]

    def tiers_for(self, gen: int) -> List[int]:
        return [int(t) for t in self.phase(gen)["tiers"]]

    def batch_for(self, gen: int) -> int:
        return int(self.phase(gen)["batch"])

    def sample_for(self, gen: int) -> List[SimpleCase]:
        phase_idx = self.phase_index(gen)
        phase = self.phases[phase_idx]
        rng = random.Random(self.seed + 1000003 * int(gen) + 97 * phase_idx)
        return sample_cases(self.cases_by_tier, phase["tiers"],
                            int(phase["batch"]), rng)

    def bundle_for(self, gen: int, total_budget: int) -> Bundle:
        phase_idx = self.phase_index(gen)
        phase = self.phases[phase_idx]
        cases = self.sample_for(gen)
        name = "curriculum-%s-g%06d" % (phase.get("name", phase_idx), gen)
        return cases_to_bundle(cases, name, total_budget, self.thresholds)

    def describe(self) -> str:
        parts = []
        for i, ph in enumerate(self.phases):
            parts.append("%d:%s(tiers=%s,gens=%d,batch=%d)"
                         % (i, ph.get("name", i), ph["tiers"],
                            int(ph["generations"]), int(ph["batch"])))
        counts = {tier: len(cases)
                  for tier, cases in sorted(self.cases_by_tier.items())}
        return ("phases=[%s] tier_counts=%s excluded_unreachable=%d seed=%d"
                % ("; ".join(parts), counts,
                   excluded_infeasible(self.thresholds), self.seed))
