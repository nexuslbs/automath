"""llm_like: the SIMPLE-ML "LLM-like" learning module (Unit D3).

This package implements the design fixed in ``docs/evolution/LLM_LIKE.md``
(Unit D1, commit on branch ``llm-like-learning``). It borrows four mechanisms
from LLMs and re-expresses them at a CPU-only, sub-300k-parameter size:

* next-action cross-entropy over DERIVED legal actions (the "next token"),
* exactly ONE attention head over the last ``K = 8`` ``(state, action)`` context
  tokens,
* a self-generated data flywheel (rollout -> solvability filter -> kept solved
  trajectories -> bounded FIFO buffer -> retrain),
* a small value head over ``(state, next-state)`` transitions trained with MSE.

It is explicitly NOT a transformer stack: no multi-head attention, no FFN, no
layer-norm stack, no pretrained weights, no text, no tokenizer. The stdlib-only
``dynamic_env`` engine and ``evolution_trainer`` package are imported, never
modified. ``torch`` is an existing training-only dependency
(``requirements-train.txt``), not a new one.

Files
-----
``features.py``  node / action / context feature builders (torch tensors).
``model.py``     :class:`LLMAgentNet` (graph encoder + 1 attention head + heads).
``flywheel.py``  rollout -> solvability filter -> kept trajectories -> buffer.
``train.py``     CE + value MSE loop, checkpoints, learning-curve CSV.
``eval.py``      held-out three-way eval, per-seed JSON.
``tests.py``     torch-gated unit checks (shapes, param count, masking, buffer).
"""

from __future__ import annotations

__all__ = ["model", "features", "flywheel", "train", "eval", "tests"]
