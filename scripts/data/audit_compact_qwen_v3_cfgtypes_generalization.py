#!/usr/bin/env python3
"""Audit a v3 release using the versioned CFG-type-only edge subcodec."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data import audit_compact_qwen_v3_generalization as audit
from scripts.data import build_compact_qwen_v3_cfgtypes_v2 as codec


def main(argv: Sequence[str] | None = None) -> int:
    previous = audit.codec
    audit.codec = codec
    try:
        return audit.main(argv)
    finally:
        audit.codec = previous


if __name__ == "__main__":
    raise SystemExit(main())
