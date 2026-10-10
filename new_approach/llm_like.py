"""Compatibility entry point named by the unit D3 dispatch.

The SIMPLE-ML llm-like implementation lives in the top-level :mod:`llm_like`
package, exactly as ``docs/evolution/LLM_LIKE.md`` section 7 fixes it. This thin
shim exists only because the unit D3 dispatch names ``new_approach/llm_like.py``;
it adds no logic, duplicates nothing, and imports ``torch`` only when a function
is actually called, so the torch-free stdlib ``new_approach`` test path is
unaffected.

Usage::

    python -m new_approach.llm_like            # run the llm_like unit tests
    python -m new_approach.llm_like --train ...  # forwarded to llm_like.train
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence


def run_tests() -> int:
    """Run the torch-gated ``llm_like`` unit checks."""
    from llm_like.tests import run

    return run()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Forward to :func:`llm_like.train.main`."""
    from llm_like.train import main as train_main

    return train_main(argv)


if __name__ == "__main__":
    sys.exit(run_tests())
