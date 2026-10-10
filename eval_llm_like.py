#!/usr/bin/env python3
"""Headless CLI entry point for the llm_like held-out evaluation (Unit D3).

Thin wrapper over :mod:`llm_like.eval` (design section 6); the module itself is
the spec-mandated implementation. Example::

    /opt/automath/venv/bin/python eval_llm_like.py \
        --seed 7 --checkpoint out/llm_like/s7/latest.pt --out out/llm_like/s7
"""

from __future__ import annotations

from llm_like.eval import main


if __name__ == "__main__":
    raise SystemExit(main())
