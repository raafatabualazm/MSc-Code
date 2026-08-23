#!/usr/bin/env python3
"""Build a v3 release with the versioned CFG-type-only edge subcodec."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data import build_compact_qwen_v3_cfgtypes_v2 as codec
from scripts.data import build_compact_qwen_v3_release as release


def main(argv: Sequence[str] | None = None) -> int:
    previous = release.codec
    release.codec = codec
    try:
        return release.main(argv)
    finally:
        release.codec = previous


if __name__ == "__main__":
    raise SystemExit(main())
