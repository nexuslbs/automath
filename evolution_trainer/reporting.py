"""Plain-text rendering of the generation table and the reward trajectory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def render_generation_summary(history: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "gen | best agent  | fitness  | steps | solved | births | deaths | subagents | best parents -> child",
        "----+-------------+----------+-------+--------+--------+--------+-----------+----------------------",
    ]
    for record in history:
        births = record["births"]
        if births:
            shown = "; ".join(
                "%s+%s->%s" % (
                    (b["parents"][0] if b["parents"] else "?"),
                    (b["parents"][1] if len(b["parents"]) > 1 else "-"),
                    b["child"],
                )
                for b in births[:2]
            )
            if len(births) > 2:
                shown += " (+%d more)" % (len(births) - 2)
        else:
            shown = "(none)"
        if len(shown) > 90:
            shown = shown[:87] + "..."
        lines.append(
            "%3d | %-11s | %8.4f | %5d | %6d | %6d | %6d | %9d | %s"
            % (record["generation"], record["best_agent"], record["best_fitness"],
               record["best_steps"], record["solved"], len(births),
               len(record["deaths"]), len(record["subagents"]), shown)
        )
    return "\n".join(lines)


def render_agent_table(history: Sequence[Dict[str, Any]],
                       generations: Optional[Sequence[int]] = None,
                       max_rows: int = 400) -> str:
    lines = [
        "gen | agent        | origin     | parents                    | fitness  | budget   | steps | solved | mut    | alive | note",
        "----+--------------+------------+----------------------------+----------+----------+-------+--------+--------+-------+-----",
    ]
    rows = 0
    for record in history:
        if generations is not None and record["generation"] not in generations:
            continue
        for row in record["agents"]:
            if rows >= max_rows:
                lines.append("... (%d rows capped)" % (len(history),))
                return "\n".join(lines)
            note = row["note"] or ""
            if len(note) > 60:
                note = note[:57] + "..."
            lines.append(
                "%3d | %-12s | %-10s | %-26s | %8.4f | %8.4f | %5d | %6d | %.4f | %5d | %s"
                % (row["generation"], row["agent_id"], row["origin"],
                   row["parents"] or "(founder)", row["fitness"], row["budget"],
                   row["steps"], row["solved"], row["mutation_magnitude"],
                   row["alive"], note)
            )
            rows += 1
    return "\n".join(lines)


def render_reward_trajectory(history: Sequence[Dict[str, Any]]) -> str:
    lines = ["gen,best_fitness,mean_fitness,solved,best_steps,births,deaths,subagents"]
    for record in history:
        lines.append("%d,%.6f,%.6f,%d,%d,%d,%d,%d" % (
            record["generation"], record["best_fitness"], record["mean_fitness"],
            record["solved"], record["best_steps"], len(record["births"]),
            len(record["deaths"]), len(record["subagents"])))
    return "\n".join(lines)


def first_birth(history: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for record in history:
        if record["births"]:
            return record["births"][0]
    return None
