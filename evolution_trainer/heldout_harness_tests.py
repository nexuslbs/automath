"""Unit tests for the standard held-out harness (task 4349 unit 5).

Run::

    /opt/automath/venv/bin/python -m evolution_trainer.heldout_harness_tests

Cheap and stdlib-only: one tiny shipped spec (``spec_minimal``), two episodes
and a short global wall cap.  No training.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile

from .heldout_harness import (
    DEFAULT_FAMILIES,
    FRESH_EPISODE_SEED,
    SizeInvariantNet,
    evaluate_case_episodes,
    evaluate_family,
    family_cases,
    find_stage_checkpoints,
    load_selected_genome,
    parse_val_term,
    run as harness_run,
    table_rows,
)
from .size_selection import SELECTION_SEED, spec_by_id


class CheckFailure(Exception):
    def __init__(self, name: str, detail: str) -> None:
        super().__init__("%s: %s" % (name, detail))
        self.name = name
        self.detail = detail


def check_parse_val_term() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "history.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("generation,agent_id,val_solved_rate\n")
            handle.write("1,a,0.0\n1,b,0.5\n2,c,0.0\n2,d,0.0\n3,e,0.25\n")
        stats = parse_val_term(path)
    if stats["generations"] != 3:
        raise CheckFailure("parse_val_term", "generations=%r" % stats["generations"])
    if stats["gen_rows_val_gt0"] != 2:
        raise CheckFailure("parse_val_term",
                           "gen_rows_val_gt0=%r" % stats["gen_rows_val_gt0"])
    if abs(stats["val_max"] - 0.5) > 1e-9 or stats["val_min"] != 0.0:
        raise CheckFailure("parse_val_term",
                           "min/max=%r/%r" % (stats["val_min"], stats["val_max"]))
    return "3 gens, 2 fired, max=%.2f" % stats["val_max"]


def check_load_selected_genome() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "stage4", "spec_x", "checkpoints")
        os.makedirs(ckpt)
        persist = os.path.join(ckpt, "best_agents_persist.json")
        with open(persist, "w", encoding="utf-8") as handle:
            json.dump({"spec_id": "spec_x",
                       "all_time_best": {"generation": 4,
                                         "agent": {"agent_id": "g003-m1",
                                                   "genome": [0.1, 0.2, 0.3]}}},
                      handle)
        seeds = find_stage_checkpoints(tmp)
        loaded = load_selected_genome(tmp)
    if len(seeds) != 1 or loaded["genome"] != [0.1, 0.2, 0.3]:
        raise CheckFailure("load_selected_genome", "genome=%r" % loaded["genome"])
    if loaded["all_time_best_generation"] != 4:
        raise CheckFailure("load_selected_genome", "generation not read")
    return "loaded all_time_best gen=4 len=3"


def check_episode_protocol() -> str:
    spec = spec_by_id("spec_minimal")
    net = SizeInvariantNet(12)
    genome = net.init(random.Random(0), scale=0.1)
    res = evaluate_case_episodes(net, genome, spec, "current", episodes=2,
                                 episode_seed=FRESH_EPISODE_SEED,
                                 mask_illegal=True, step_budget=12)
    if not (0 <= res["solved"] <= 2):
        raise CheckFailure("episode_protocol", "solved=%r" % res["solved"])
    if not (0 < res["step_budget"] <= 12):
        raise CheckFailure("episode_protocol", "step_budget=%r" % res["step_budget"])
    if len(set(res["episode_seeds"])) != 2:
        raise CheckFailure("episode_protocol", "episode seeds not distinct")
    if res["episode_seeds"][0] == SELECTION_SEED:
        raise CheckFailure("episode_protocol", "episode seed equals recorded seed")
    return "episodes=2 solved=%d rate=%s" % (res["solved"], res["mean_solve_rate"])


def check_three_way_sums() -> str:
    net = SizeInvariantNet(12)
    genome = net.init(random.Random(1), scale=0.1)
    out = evaluate_family(net, genome, "spec_minimal", "current", episodes=2,
                          episode_seed=FRESH_EPISODE_SEED, step_budget=12,
                          max_wall_secs=60.0)
    total = (out["SOLVABLE-SOLVED"] + out["SOLVABLE-UNSOLVED"]
             + out["PROVABLY-UNSOLVABLE"] + out["SKIPPED-BUDGET"])
    if total != out["n"]:
        raise CheckFailure("three_way_sums", "buckets %d != n %d" % (total, out["n"]))
    json.dumps(out)
    return "n=%d buckets=%d" % (out["n"], total)


def check_run_and_table() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "stage4", "spec_minimal", "checkpoints")
        hist = os.path.join(tmp, "stage4", "spec_minimal", "history")
        os.makedirs(ckpt)
        os.makedirs(hist)
        net = SizeInvariantNet(12)
        genome = net.init(random.Random(2), scale=0.1)
        with open(os.path.join(ckpt, "best_agents_persist.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"spec_id": "spec_minimal",
                       "all_time_best": {"generation": 1,
                                         "agent": {"agent_id": "g000",
                                                   "genome": genome}}}, handle)
        with open(os.path.join(hist, "history.csv"), "w",
                  encoding="utf-8") as handle:
            handle.write("generation,agent_id,val_solved_rate\n1,a,0.5\n")
        payload = harness_run([tmp], families=["spec_minimal"],
                      action_sets=["current"], episodes=2,
                      episode_seed=FRESH_EPISODE_SEED, step_budget=12,
                      max_wall_secs=60.0)
        lines = table_rows(payload)
    if len(lines) < 3:
        raise CheckFailure("run_and_table", "table lines=%d" % len(lines))
    if payload["seeds"][0]["selected_val_solved_rate"] != 0.5:
        raise CheckFailure("run_and_table",
                           "selected val=%r"
                           % payload["seeds"][0]["selected_val_solved_rate"])
    return "table %d lines, selected val=0.5" % len(lines)


def check_spec_case_adapter() -> str:
    from . import solvability as oracle

    cases = family_cases("spec_multi_step")
    name, case = cases[0]
    res = oracle.classify(case, "current")
    if res["verdict"] != oracle.SOLVABLE:
        raise CheckFailure("spec_case_adapter", "verdict=%r" % res["verdict"])
    if res.get("proof_method") != "provided_witness_replay":
        raise CheckFailure("spec_case_adapter",
                           "proof=%r" % res.get("proof_method"))
    return "spec case classified %s via %s" % (res["verdict"],
                                               res["proof_method"])


CHECKS = [
    ("parse_val_term", check_parse_val_term),
    ("load_selected_genome", check_load_selected_genome),
    ("episode_protocol", check_episode_protocol),
    ("three_way_sums", check_three_way_sums),
    ("spec_case_adapter", check_spec_case_adapter),
    ("run_and_table", check_run_and_table),
]


def run() -> int:
    passed = 0
    failures = []
    for name, fn in CHECKS:
        try:
            detail = fn()
        except CheckFailure as exc:
            failures.append(exc.name)
            print("[FAIL] %s: %s" % (name, exc.detail))
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print("[FAIL] %s: unexpected %r" % (name, exc))
        else:
            passed += 1
            print("[PASS] %s: %s" % (name, detail))
    print("RESULT: %d/%d passed (held-out harness)" % (passed, len(CHECKS)))
    return 0 if not failures else 1


def test_parse_val_term() -> None:
    check_parse_val_term()


def test_load_selected_genome() -> None:
    check_load_selected_genome()


def test_episode_protocol() -> None:
    check_episode_protocol()


def test_three_way_sums() -> None:
    check_three_way_sums()


def test_spec_case_adapter() -> None:
    check_spec_case_adapter()


def test_run_and_table() -> None:
    check_run_and_table()


if __name__ == "__main__":
    sys.exit(run())
