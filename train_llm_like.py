#!/usr/bin/env python3
"""Headless CLI entry point for the llm_like data flywheel (Unit D3).

Thin wrapper over :mod:`llm_like.train` (design section 5); the module itself is
the spec-mandated implementation. Example::

    /opt/automath/venv/bin/python train_llm_like.py \
        --seed 7 --rounds 8 --d 32 --out out/llm_like/s7
"""

from __future__ import annotations

from llm_like.train import main


if __name__ == "__main__":
    raise SystemExit(main())
