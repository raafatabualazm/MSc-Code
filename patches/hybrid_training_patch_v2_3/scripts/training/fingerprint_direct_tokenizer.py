#!/usr/bin/env python3
"""Print the logical tokenizer fingerprint used by direct compact contracts."""

from __future__ import annotations

import argparse
import json

from models.direct_compact_causal import tokenizer_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer exposes neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    print(
        json.dumps(
            {
                "tokenizer": args.tokenizer,
                "revision": args.revision,
                "tokenizer_fingerprint_sha256": tokenizer_fingerprint(tokenizer),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
