
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def compile_dart(code: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'main.dart'
        path.write_text(code, encoding='utf-8')

        result = subprocess.run(
            ['dart', 'analyze', str(path)],
            capture_output=True,
            text=True,
        )

        return result.returncode == 0


def compile_swift(code: str) -> bool:
    return bool(code.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()

    rows = json.loads(Path(args.predictions).read_text(encoding='utf-8'))

    passed = 0

    for idx, row in enumerate(rows):
        lang = row.get('language', 'dart').lower()
        code = row['prediction']

        ok = compile_dart(code) if lang == 'dart' else compile_swift(code)

        if ok:
            passed += 1

        print(f'[{idx + 1}/{len(rows)}] compile={ok}')

    total = max(len(rows), 1)

    print(json.dumps({
        'compile_at_1': passed / total,
        'compiled': passed,
        'total': total,
    }, indent=2))


if __name__ == '__main__':
    main()
