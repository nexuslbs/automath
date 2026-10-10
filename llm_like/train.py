"""Training loop for llm_like (design section 4 + the section 5 flywheel).

Objective ``L = L_policy + lambda_v * L_value`` with ``lambda_v = 0.5``:

* ``L_policy`` next-action cross-entropy over the LEGAL actions only (illegal
  actions never enter the softmax; the label is the action the kept solved
  episode actually took),
* ``L_value`` MSE of ``v(s, s')`` against the normalized return-to-go of the
  solved trajectory (or the normalized BFS distance-to-goal when a witness is
  available).

Optimizer Adam, ``lr = 1e-3``, no weight decay; mini-batches of 256 step
records; one pass over the bounded FIFO buffer per flywheel round. The context
tokens are recomputed from the stored per-step features under ``no_grad`` each
round (so the attention "context" is an input, not a stale cached activation).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .flywheel import (
    Case,
    FifoStepBuffer,
    HELDOUT_SPEC_IDS,
    MAX_BUFFER_STEPS,
    filter_pool,
    load_spec_by_id,
    return_to_go,
    rollout_episode,
    solve_rate,
    training_cases,
)
from .eval import evaluate_model, heldout_cases
from .model import DEFAULT_CONTEXT, DEFAULT_WIDTH, LLMAgentNet, build_model

CURVE_FIELDS = (
    "round",
    "cumulative_episodes",
    "round_episodes",
    "round_solved",
    "buffer_steps",
    "episodes_in_buffer",
    "loss_policy",
    "loss_value",
    "loss_total",
    "solution_rate_mask_off",
    "solution_rate_mask_on",
    "mean_steps_mask_off",
    "mean_steps_mask_on",
    "wall_s",
)


@dataclass
class TrainConfig:
    seed: int = 7
    rounds: int = 8
    d: int = DEFAULT_WIDTH
    k_context: int = DEFAULT_CONTEXT
    batch: int = 256
    lr: float = 1e-3
    lambda_v: float = 0.5
    epsilon: float = 0.1
    max_buffer_steps: int = MAX_BUFFER_STEPS
    action_set: str = "current"
    include_val64: bool = False
    include_heldout: bool = False
    max_cases: int = 0            # 0 = all training-pool cases
    eval_every: int = 1
    eval_limit: int = 0           # 0 = full held-out set
    eval_val_size: int = 64
    eval_fresh_size: int = 64
    eval_perturb: int = 30
    checkpoint_every: int = 1
    patience: int = 3             # plateau stop (design 5.1 step 8)
    seed_witness: bool = True     # optional one-time BFS-witness buffer seed
    out: str = "out/llm_like"


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # pragma: no cover - older torch / unsupported op
        pass


# --------------------------------------------------------------------------
# Context cache (one no_grad pass per episode per round)
# --------------------------------------------------------------------------

def episode_contexts(net: LLMAgentNet,
                     records: Sequence[Dict[str, Any]]) -> List[torch.Tensor]:
    """Recompute the detached attention context for every step of an episode."""
    states: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    contexts: List[torch.Tensor] = []
    with torch.no_grad():
        for record in records:
            pooled, node_latent = net.encode_graph(
                record["node_features"], record["type_idx"], record["adj"])
            action_latent = net.action_latents(
                record["action_features"], record["action_ref_local"],
                node_latent)
            tokens, mask = net.context_tokens(states, actions, pooled)
            contexts.append(net.attend(tokens, mask))
            states.append(pooled)
            actions.append(action_latent[int(record["chosen"])])
    return contexts


def train_pass(net: LLMAgentNet, optimizer: torch.optim.Optimizer,
               buffer: FifoStepBuffer, cfg: TrainConfig) -> Dict[str, float]:
    """One pass over the bounded buffer in mini-batches of ``cfg.batch`` steps."""
    net.train()
    pending: List[Tuple[torch.Tensor, float, float]] = []
    totals = {"policy": 0.0, "value": 0.0, "total": 0.0, "steps": 0, "batches": 0}

    def flush() -> None:
        if not pending:
            return
        optimizer.zero_grad()
        loss = sum(item[0] for item in pending)
        loss.backward()
        optimizer.step()
        totals["policy"] += sum(item[1] for item in pending)
        totals["value"] += sum(item[2] for item in pending)
        totals["total"] += float(loss.item())
        totals["steps"] += len(pending)
        totals["batches"] += 1
        pending.clear()

    for episode in buffer.episodes:
        records = episode.get("records", ())
        if not records:
            continue
        contexts = episode_contexts(net, records)
        for record, context_vec in zip(records, contexts):
            pooled, node_latent = net.encode_graph(
                record["node_features"], record["type_idx"], record["adj"])
            action_latent = net.action_latents(
                record["action_features"], record["action_ref_local"],
                node_latent)
            logits = net.policy_logits(context_vec, action_latent)
            target = torch.tensor([int(record["chosen"])], dtype=torch.long)
            loss_policy = nn.functional.cross_entropy(
                logits.unsqueeze(0), target)
            next_pooled, _ = net.encode_graph(
                record["next_node_features"], record["next_type_idx"],
                record["next_adj"])
            value = net.value(pooled, next_pooled)
            goal = torch.tensor(float(record["return_to_go"]))
            loss_value = (value - goal) ** 2
            loss = loss_policy + float(cfg.lambda_v) * loss_value
            pending.append((loss, float(loss_policy.item()),
                            float(loss_value.item())))
            if len(pending) >= int(cfg.batch):
                flush()
    flush()

    steps = max(1, totals["steps"])
    return {
        "loss_policy": totals["policy"] / steps,
        "loss_value": totals["value"] / steps,
        "loss_total": totals["total"] / max(1, totals["batches"]),
        "batches": float(totals["batches"]),
        "steps": float(totals["steps"]),
    }


# --------------------------------------------------------------------------
# Optional one-time BFS-witness seed (design 5.1, never an ongoing signal)
# --------------------------------------------------------------------------

def _witness_for(case: Case) -> Optional[Tuple[str, ...]]:
    if case.witness is not None:
        return tuple(case.witness)
    try:
        from evolution_trainer.curriculum import bfs_min_steps

        _minimum, word = bfs_min_steps(case.spec)
        return tuple(word) if word else None
    except Exception:
        return None


def seed_buffer_with_witness(net: LLMAgentNet, buffer: FifoStepBuffer,
                             cases: Sequence[Case],
                             allow_pop: bool) -> int:
    added = 0
    for index, case in enumerate(cases):
        witness = _witness_for(case)
        if not witness:
            continue
        episode = rollout_episode(net, case, random.Random(0), epsilon=0.0,
                                  mask_illegal=False, allow_pop=allow_pop,
                                  max_steps=case.step_budget(),
                                  round_index=-1, episode_index=index,
                                  scripted_keys=witness)
        if episode["solved"]:
            buffer.add_episode(episode)
            added += 1
    return added


# --------------------------------------------------------------------------
# Checkpoints + learning curve
# --------------------------------------------------------------------------

def save_checkpoint(net: LLMAgentNet, out_dir: str, round_index: int,
                    meta: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "checkpoint_round_%05d.pt" % round_index)
    torch.save(net.state_dict(), path)
    torch.save(net.state_dict(), os.path.join(out_dir, "latest.pt"))
    payload = dict(meta)
    payload.update({
        "round": round_index,
        "param_count": net.param_count(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    with open(os.path.join(out_dir, "checkpoint.json"), "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def write_curve_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(CURVE_FIELDS) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(field, ""))
                                  for field in CURVE_FIELDS) + "\n")


# --------------------------------------------------------------------------
# The flywheel
# --------------------------------------------------------------------------

def train(cfg: TrainConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    net = build_model(cfg.d, cfg.k_context, seed=cfg.seed)
    print("[llm_like] ACTUAL parameter count = %d (target <= 300000)"
          % net.param_count())
    print("[llm_like] parameter breakdown = %s"
          % json.dumps(net.param_breakdown(), sort_keys=True))

    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg.lr))
    buffer = FifoStepBuffer(cfg.max_buffer_steps)
    allow_pop = cfg.action_set == "pop"

    pool_cases = training_cases(include_val64=cfg.include_val64,
                                include_heldout=cfg.include_heldout)
    filtered = filter_pool(pool_cases, action_set=cfg.action_set)
    pool = list(filtered["active"])
    if cfg.max_cases and cfg.max_cases > 0:
        pool = pool[: int(cfg.max_cases)]
    print("[llm_like] training pool=%d (excluded UNSOLVABLE=%d) counts=%s"
          % (len(pool), len(filtered["excluded"]), filtered["counts"]))

    eval_cases = heldout_cases(val_size=cfg.eval_val_size,
                               fresh_size=cfg.eval_fresh_size,
                               perturb_count=cfg.eval_perturb,
                               limit=(cfg.eval_limit or None))
    eval_filtered = filter_pool(eval_cases, action_set=cfg.action_set) \
        if eval_cases else None
    if eval_filtered is not None:
        print("[llm_like] held-out set=%d three_way=%s"
              % (len(eval_cases), eval_filtered["counts"]))

    seeded = (seed_buffer_with_witness(net, buffer, pool, allow_pop)
              if cfg.seed_witness else 0)
    print("[llm_like] one-time BFS-witness seed episodes=%d buffer_steps=%d"
          % (seeded, len(buffer)))

    rng = random.Random(cfg.seed)
    cumulative_episodes = 0
    curve: List[Dict[str, Any]] = []
    best_rate = -1.0
    plateau = 0
    started = time.perf_counter()
    out_dir = os.path.abspath(cfg.out)
    os.makedirs(out_dir, exist_ok=True)

    for round_index in range(int(cfg.rounds)):
        covered = buffer.covered_forms()
        new_forms = [case for case in pool if case.form_key() not in covered]
        rollout_pool = new_forms if new_forms else list(pool)
        round_episodes = 0
        round_solved = 0
        for index, case in enumerate(rollout_pool):
            episode = rollout_episode(
                net, case, rng, epsilon=float(cfg.epsilon),
                mask_illegal=False, allow_pop=allow_pop,
                max_steps=case.step_budget(), round_index=round_index,
                episode_index=index)
            round_episodes += 1
            cumulative_episodes += 1
            if episode["solved"]:
                round_solved += 1
                buffer.add_episode(episode)
        losses = train_pass(net, optimizer, buffer, cfg)

        do_eval = (round_index % int(cfg.eval_every) == 0
                   or round_index == int(cfg.rounds) - 1)
        eval_metrics: Dict[str, Any] = {}
        if do_eval and eval_filtered is not None and eval_cases:
            eval_metrics = evaluate_model(
                net, eval_cases, action_set=cfg.action_set, allow_pop=allow_pop,
                seed=cfg.seed, filtered=eval_filtered)
        off = (eval_metrics.get("mask_off") or {}) if eval_metrics else {}
        on = (eval_metrics.get("mask_on") or {}) if eval_metrics else {}
        row = {
            "round": round_index,
            "cumulative_episodes": cumulative_episodes,
            "round_episodes": round_episodes,
            "round_solved": round_solved,
            "buffer_steps": len(buffer),
            "episodes_in_buffer": buffer.stats()["episodes"],
            "loss_policy": round(losses["loss_policy"], 6),
            "loss_value": round(losses["loss_value"], 6),
            "loss_total": round(losses["loss_total"], 6),
            "solution_rate_mask_off": off.get("solution_rate"),
            "solution_rate_mask_on": on.get("solution_rate"),
            "mean_steps_mask_off": off.get("mean_steps_solved"),
            "mean_steps_mask_on": on.get("mean_steps_solved"),
            "wall_s": round(time.perf_counter() - started, 4),
        }
        curve.append(row)
        write_curve_csv(os.path.join(out_dir, "learning_curve.csv"), curve)
        print("[llm_like] round=%d episodes=%d solved=%d buffer=%d "
              "loss_p=%.4f loss_v=%.4f rate_off=%s rate_on=%s steps_off=%s"
              % (round_index, round_episodes, round_solved, len(buffer),
                 losses["loss_policy"], losses["loss_value"],
                 row["solution_rate_mask_off"], row["solution_rate_mask_on"],
                 row["mean_steps_mask_off"]))

        if cfg.checkpoint_every and (round_index % int(cfg.checkpoint_every) == 0):
            save_checkpoint(net, out_dir, round_index, {
                "seed": cfg.seed, "config": asdict(cfg),
                "buffer_stats": buffer.stats(), "curve_point": row,
            })

        rate = row["solution_rate_mask_off"]
        if rate is not None and float(rate) > best_rate + 1e-9:
            best_rate = float(rate)
            plateau = 0
        else:
            plateau += 1
        if plateau >= int(cfg.patience):
            print("[llm_like] plateau stop after %d rounds (P=%d)"
                  % (round_index + 1, cfg.patience))
            break

    save_checkpoint(net, out_dir, len(curve) - 1, {
        "seed": cfg.seed, "config": asdict(cfg), "buffer_stats": buffer.stats(),
        "final": True,
    })
    summary = {
        "seed": cfg.seed,
        "param_count": net.param_count(),
        "param_breakdown": net.param_breakdown(),
        "config": asdict(cfg),
        "rounds_run": len(curve),
        "cumulative_episodes": cumulative_episodes,
        "training_pool": len(pool),
        "training_filter_counts": filtered["counts"],
        "buffer_stats": buffer.stats(),
        "curve": curve,
        "elapsed_s": round(time.perf_counter() - started, 4),
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "DONE"), "w", encoding="utf-8") as handle:
        handle.write("seed=%s rounds=%d param_count=%d\n"
                     % (cfg.seed, len(curve), net.param_count()))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="llm_like flywheel training")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--d", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-v", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max-buffer-steps", type=int, default=MAX_BUFFER_STEPS)
    parser.add_argument("--action-set", default="current",
                        choices=["current", "pop"])
    parser.add_argument("--include-val64", action="store_true")
    parser.add_argument("--include-heldout", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-fresh-size", type=int, default=64)
    parser.add_argument("--eval-val-size", type=int, default=64)
    parser.add_argument("--eval-perturb", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--no-witness-seed", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        seed=args.seed, rounds=args.rounds, d=args.d, k_context=args.context,
        batch=args.batch, lr=args.lr, lambda_v=args.lambda_v,
        epsilon=args.epsilon, max_buffer_steps=args.max_buffer_steps,
        action_set=args.action_set, include_val64=args.include_val64,
        include_heldout=args.include_heldout, max_cases=args.max_cases,
        eval_every=args.eval_every, eval_limit=args.eval_limit,
        eval_val_size=args.eval_val_size, eval_fresh_size=args.eval_fresh_size,
        eval_perturb=args.eval_perturb,
        checkpoint_every=args.checkpoint_every, patience=args.patience,
        seed_witness=not args.no_witness_seed, out=args.out)
    summary = train(cfg)
    print("[llm_like] DONE rounds=%d param_count=%d out=%s"
          % (summary["rounds_run"], summary["param_count"], summary["out_dir"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
