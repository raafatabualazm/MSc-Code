#!/usr/bin/env python3
"""Canonical entry point for the sealed OpenAI GPT-5.6 Batch ceiling.

The implementation lives in :mod:`openai56_batch_fasttrack` so the original
development filename remains import-compatible with its focused tests and any
already-written preflight artifacts.
"""
from __future__ import annotations

from openai56_batch_fasttrack import main


if __name__ == "__main__":
    raise SystemExit(main())
